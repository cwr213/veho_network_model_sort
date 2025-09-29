# veho_net/milp_sort.py - FIXED: Multi-level sortation optimization with proper minimum capacity logic
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from .time_cost_sort import weighted_pkg_cube, calculate_truck_capacity
from .geo_sort import haversine_miles, band_lookup
from .sort_optimization import (
    build_facility_relationships, calculate_sort_point_requirements,
    get_sort_level_options, calculate_sort_level_costs, validate_sort_capacity_constraints
)


def _arc_cost_per_truck(u: str, v: str, facilities: pd.DataFrame, mileage_bands: pd.DataFrame) -> Tuple[
    float, float, float]:
    """Calculate distance and transportation cost for arc between two facilities."""
    fac = facilities.set_index("facility_name")[["lat", "lon"]].astype(float)
    lat1, lon1 = fac.at[u, "lat"], fac.at[u, "lon"]
    lat2, lon2 = fac.at[v, "lat"], fac.at[v, "lon"]
    raw = haversine_miles(lat1, lon1, lat2, lon2)
    fixed, var, circuit, mph = band_lookup(raw, mileage_bands)
    dist = raw * circuit
    return dist, fixed + var * dist, mph


def _legs_for_candidate(row: pd.Series, facilities: pd.DataFrame, mileage_bands: pd.DataFrame):
    """Extract leg information from candidate path for arc analysis."""
    nodes = row.get("path_nodes", None)
    if isinstance(nodes, list) and len(nodes) >= 2:
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = []
        for u, v in pairs:
            dist, cost_per_truck, mph = _arc_cost_per_truck(u, v, facilities, mileage_bands)
            legs.append((u, v, dist, cost_per_truck, mph))
        return legs

    # Fallback to basic origin->dest
    o, d = row["origin"], row["dest"]
    dist, cost_per_truck, mph = _arc_cost_per_truck(o, d, facilities, mileage_bands)
    return [(o, d, dist, cost_per_truck, mph)]


def calculate_facility_sort_requirements(facility_name, groups, sort_decision, relationships, facility_lookup,
                                         sort_points_per_dest):
    """
    Calculate sort point requirements with proper minimum calculation.

    Minimum sort points = (own_region_destinations + external_regions_count) × sort_points_per_dest

    where:
    - own_region_destinations = children_count + (1 if hybrid else 0)
    - external_regions_count = unique_regional_sort_hubs_served - 1

    Returns:
        Dict with sort point requirements breakdown including minimum calculation
    """
    regional_sort_hub_map = relationships['regional_sort_hub_map']
    facility_types = relationships['facility_types']

    # STEP 1: Identify all destinations this facility serves as injection hub
    injection_destinations = set()
    for group_name, group_idxs in groups.items():
        scenario_id, origin, dest, day_type = group_name
        if origin == facility_name:
            injection_destinations.add(dest)

    # STEP 2: Identify child facilities using regional_sort_hub
    child_facilities = set()
    for child_name in facility_lookup.index:
        child_regional_sort_hub = regional_sort_hub_map.get(child_name, child_name)
        if child_regional_sort_hub == facility_name and child_name != facility_name:
            child_facilities.add(child_name)

    # STEP 3: Calculate MINIMUM sort points needed (optimal region-level sorting)
    # Note: This assumes optimal region-level for own region and external regions
    # Sort group level would require MORE than this minimum

    # Own region destinations = children + self (if hybrid)
    own_region_count = len(child_facilities)
    facility_type = facility_types.get(facility_name, '').lower()
    if facility_type == 'hybrid':
        own_region_count += 1  # Add self-destination for hybrid facilities

    # External regions = unique regional sort hubs served (excluding own region)
    unique_regional_sort_hubs = set()
    for dest in injection_destinations:
        dest_regional_sort_hub = regional_sort_hub_map.get(dest, dest)
        unique_regional_sort_hubs.add(dest_regional_sort_hub)

    # Remove own region from external count
    external_regions_count = len(unique_regional_sort_hubs)
    if facility_name in unique_regional_sort_hubs:
        external_regions_count -= 1

    # Calculate theoretical minimum (assumes optimal region-level consolidation)
    # and market-level sort points needed
    minimum_sort_points = (own_region_count + external_regions_count) * sort_points_per_dest
    market_level_sort_points = len(injection_destinations) * sort_points_per_dest

    # Note: Sort group level would require even more than market level due to multipliers

    return {
        'injection_destinations_count': len(injection_destinations),
        'minimum_sort_points': minimum_sort_points,
        'market_level_sort_points': market_level_sort_points,
        'includes_self_destination': facility_name in injection_destinations,
        'child_facilities_count': len(child_facilities),
        'child_facilities': child_facilities,
        'injection_destinations': injection_destinations,
        'own_region_count': own_region_count,
        'external_regions_count': external_regions_count,
        'unique_regional_sort_hubs_count': len(unique_regional_sort_hubs),
        'unique_regional_sort_hubs': unique_regional_sort_hubs
    }


def solve_arc_pooled_path_selection_with_sort_optimization(
        candidates: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        cost_kv: Dict[str, float],
        timing_kv: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    UPDATED ARCHITECTURE: Path selection with intelligent sort capacity management.

    Returns:
        - od_selected: Optimal paths with sort level decisions
        - arc_summary: Arc/lane utilization
        - network_kpis: Network performance metrics
        - sort_summary: Sort level decision analysis
    """
    cand = candidates.reset_index(drop=True).copy()
    strategy = str(cost_kv.get("load_strategy", "container")).lower()

    print(f"    UPDATED MILP: Path selection with intelligent sort capacity management")
    print(f"    Strategy: {strategy}")
    print(f"    Candidate paths: {len(cand)}")

    # Build facility relationships for sort optimization
    relationships = build_facility_relationships(facilities)
    hub_facilities = relationships['hub_facilities']

    # Get timing/cost parameters for sort optimization
    sort_points_per_dest = float(timing_kv.get('sort_points_per_destination', 1.0))

    # Create facility lookup for capacities and sort groups
    facility_lookup = facilities.set_index('facility_name')

    path_keys = list(cand.index)
    w_cube = weighted_pkg_cube(package_mix)

    # Build arc metadata and path-to-arc mapping
    arc_index_map: Dict[Tuple[str, str], int] = {}
    arc_meta: List[Dict] = []
    path_arcs: Dict[int, List[int]] = {}
    path_od_data: Dict[int, Dict] = {}

    for i in path_keys:
        r = cand.loc[i]
        legs = _legs_for_candidate(r, facilities, mileage_bands)

        path_nodes = r.get("path_nodes", [r["origin"], r["dest"]])
        if not isinstance(path_nodes, list):
            path_nodes = [r["origin"], r["dest"]]

        # Store path metadata for sort optimization
        path_od_data[i] = {
            'origin': r["origin"],
            'dest': r["dest"],
            'pkgs_day': float(r["pkgs_day"]),
            'scenario_id': r.get("scenario_id", "default"),
            'day_type': r.get("day_type", "peak"),
            'path_str': r.get("path_str", f"{r['origin']}->{r['dest']}"),
            'path_type': r.get("path_type", "direct"),
            'path_nodes': path_nodes
        }

        # Map path to arcs
        ids = []
        for (u, v, dist, cost_per_truck, mph) in legs:
            key = (u, v)
            if key not in arc_index_map:
                arc_index_map[key] = len(arc_meta)
                arc_meta.append({
                    "from": u,
                    "to": v,
                    "distance_miles": dist,
                    "cost_per_truck": cost_per_truck,
                    "mph": mph
                })
            ids.append(arc_index_map[key])

        path_arcs[i] = ids

    print(f"    Generated {len(arc_meta)} unique arcs from {len(cand)} candidate paths")

    # Initialize CP-SAT optimization model
    model = cp_model.CpModel()

    # Path selection variables: choose exactly one path per OD group
    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}

    for group_name, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    print(f"    Created {len(groups)} OD selection constraints")

    # Sort level decision variables for each OD group
    sort_levels = ['region', 'market', 'sort_group']
    od_groups = list(groups.keys())

    # Create sort level decision variables
    sort_decision = {}
    for group_name in od_groups:
        scenario_id, origin, dest, day_type = group_name

        # Get valid sort options for this OD (use first path in group as representative)
        repr_path_idx = groups[group_name][0]
        repr_od_row = cand.loc[repr_path_idx]

        valid_options = get_sort_level_options(repr_od_row, facilities)

        # Create binary variables for each valid sort level
        group_sort_vars = {}
        for sort_level in sort_levels:
            if sort_level in valid_options:
                group_sort_vars[sort_level] = model.NewBoolVar(
                    f"sort_{scenario_id}_{origin}_{dest}_{day_type}_{sort_level}")
            else:
                group_sort_vars[sort_level] = None

        sort_decision[group_name] = group_sort_vars

        # Constraint: exactly one sort level must be chosen
        valid_vars = [var for var in group_sort_vars.values() if var is not None]
        if valid_vars:
            model.Add(sum(valid_vars) == 1)

    print(f"    Created sort level decision variables for {len(od_groups)} OD groups")

    # Sort point capacity constraints with intelligent minimum calculation
    facility_sort_points = {}

    # Create capacity constraints for ALL facilities that could consume sort points
    all_relevant_facilities = set(hub_facilities)

    # Add parent hubs to the list
    for facility_name in facilities['facility_name']:
        if facility_name in facility_lookup.index:
            parent_hub = relationships['regional_sort_hub_map'][facility_name]
            if parent_hub in facility_lookup.index and facility_lookup.at[parent_hub, 'type'] in ['hub', 'hybrid']:
                all_relevant_facilities.add(parent_hub)

    for facility_name in all_relevant_facilities:
        if facility_name not in facility_lookup.index:
            continue

        # Get max capacity
        max_capacity = facility_lookup.at[facility_name, 'max_sort_points_capacity']

        if pd.isna(max_capacity) or max_capacity <= 0:
            print(f"    WARNING: {facility_name} missing capacity - setting to 1000 for now")
            max_capacity = 1000
        else:
            max_capacity = int(max_capacity)

        # Calculate sort requirements using UPDATED logic
        requirements = calculate_facility_sort_requirements(
            facility_name, groups, sort_decision, relationships,
            facility_lookup, sort_points_per_dest
        )

        # Create sort point usage variable
        facility_sort_points[facility_name] = model.NewIntVar(0, max_capacity, f"sort_points_{facility_name}")

        # CORRECTED: Build constraint terms with proper sort group capacity accounting
        sort_point_terms = []

        # Check feasibility first
        if max_capacity < requirements['minimum_sort_points']:
            print(
                f"      ❌ INFEASIBLE: {facility_name} needs minimum {requirements['minimum_sort_points']} but only has {max_capacity}")
            print(f"         This facility cannot operate even with optimal region-level sorting")
            print(f"         Consider increasing max_sort_points_capacity for {facility_name}")

        if requirements['injection_destinations_count'] > 0:

            # Create destination-level sort decision variables
            dest_sort_decisions = {}
            for dest in requirements['injection_destinations']:
                dest_sort_decisions[dest] = {}

                # Find sample group to get valid sort levels
                sample_group_name = None
                for group_name, group_idxs in groups.items():
                    scenario_id, origin, dest_check, day_type = group_name
                    if origin == facility_name and dest_check == dest:
                        sample_group_name = group_name
                        break

                if sample_group_name:
                    sample_sort_vars = sort_decision[sample_group_name]

                    # Create destination-level sort variables
                    for sort_level in ['region', 'market', 'sort_group']:
                        if sample_sort_vars[sort_level] is not None:
                            dest_sort_decisions[dest][sort_level] = model.NewBoolVar(
                                f"dest_sort_{facility_name}_{dest}_{sort_level}")
                        else:
                            dest_sort_decisions[dest][sort_level] = None

                    # Constraint: exactly one sort level per destination
                    valid_dest_vars = [var for var in dest_sort_decisions[dest].values() if var is not None]
                    if valid_dest_vars:
                        model.Add(sum(valid_dest_vars) == 1)

            # Link OD-level decisions to destination-level decisions
            for group_name, group_idxs in groups.items():
                scenario_id, origin, dest, day_type = group_name
                if origin == facility_name and dest in requirements['injection_destinations']:
                    od_sort_vars = sort_decision[group_name]
                    dest_sort_vars = dest_sort_decisions[dest]

                    for sort_level in ['region', 'market', 'sort_group']:
                        if od_sort_vars[sort_level] is not None and dest_sort_vars[sort_level] is not None:
                            model.Add(od_sort_vars[sort_level] == dest_sort_vars[sort_level])

            # STEP 1: Handle SORT GROUP level destinations individually (highest capacity requirement)
            for dest in requirements['injection_destinations']:
                if dest_sort_decisions[dest]['sort_group'] is not None:
                    if dest in facility_lookup.index:
                        sort_groups_count = facility_lookup.at[dest, 'last_mile_sort_groups_count']
                        if pd.isna(sort_groups_count) or sort_groups_count <= 0:
                            sort_groups_count = 4  # Default fallback
                        else:
                            sort_groups_count = int(sort_groups_count)

                        # Sort group level: multiplied sort points per destination
                        sort_group_points = int(sort_points_per_dest * sort_groups_count)
                        sort_point_terms.append(dest_sort_decisions[dest]['sort_group'] * sort_group_points)

            # STEP 2: Handle REGION/MARKET level destinations with regional consolidation
            # Group destinations by regional sort hub (excluding sort_group destinations)
            destinations_by_regional_hub = {}
            for dest in requirements['injection_destinations']:
                dest_regional_sort_hub = relationships['regional_sort_hub_map'].get(dest, dest)
                if dest_regional_sort_hub not in destinations_by_regional_hub:
                    destinations_by_regional_hub[dest_regional_sort_hub] = []
                destinations_by_regional_hub[dest_regional_sort_hub].append(dest)

            # Apply regional consolidation logic to region/market destinations only
            for regional_hub, destinations_in_hub in destinations_by_regional_hub.items():

                if regional_hub == facility_name:
                    # OWN REGION: Can use regional consolidation for capacity relief

                    # Separate region vs market destinations in own region
                    own_region_region_terms = []
                    own_region_market_terms = []

                    for dest in destinations_in_hub:
                        if dest_sort_decisions[dest]['region'] is not None:
                            own_region_region_terms.append(dest_sort_decisions[dest]['region'])
                        if dest_sort_decisions[dest]['market'] is not None:
                            own_region_market_terms.append(dest_sort_decisions[dest]['market'])

                    # If ANY destination in own region uses region-level, consolidate to 1 sort point
                    if own_region_region_terms:
                        any_region_level = model.NewBoolVar(f"any_region_{facility_name}")

                        # any_region_level = 1 if ANY destination uses region-level sorting
                        model.Add(any_region_level <= sum(own_region_region_terms))
                        for term in own_region_region_terms:
                            model.Add(any_region_level >= term)

                        # Regional consolidation: 1 sort point for entire own region when region-level used
                        sort_point_terms.append(any_region_level * int(sort_points_per_dest))

                    # Market level destinations in own region: individual sort points
                    if own_region_market_terms:
                        for market_term in own_region_market_terms:
                            sort_point_terms.append(market_term * int(sort_points_per_dest))

                else:
                    # EXTERNAL REGION: Always 1 sort point per regional hub
                    external_region_active = model.NewBoolVar(f"external_region_{facility_name}_{regional_hub}")

                    # Active if ANY destination in this external region is served (region or market level)
                    external_terms = []
                    for dest in destinations_in_hub:
                        if dest_sort_decisions[dest]['region'] is not None:
                            external_terms.append(dest_sort_decisions[dest]['region'])
                        if dest_sort_decisions[dest]['market'] is not None:
                            external_terms.append(dest_sort_decisions[dest]['market'])

                    if external_terms:
                        # External region active if any destination served
                        model.Add(external_region_active <= sum(external_terms))
                        for term in external_terms:
                            model.Add(external_region_active >= term)

                        # 1 sort point per external regional hub
                        sort_point_terms.append(external_region_active * int(sort_points_per_dest))

        # Add capacity constraint
        if sort_point_terms:
            model.Add(facility_sort_points[facility_name] >= sum(sort_point_terms))
            model.Add(facility_sort_points[facility_name] <= max_capacity)
        else:
            model.Add(facility_sort_points[facility_name] == 0)

        # Enhanced debug output with minimum sort points analysis
        role_desc = []
        if requirements['injection_destinations_count'] > 0:
            role_desc.append(f"injection({requirements['injection_destinations_count']} destinations)")
        if requirements['child_facilities_count'] > 0:
            role_desc.append(f"parent_hub({requirements['child_facilities_count']} children)")

        print(f"    {facility_name} ({' + '.join(role_desc)}): {sort_points_per_dest} sort_points/dest")
        print(f"      Available capacity: {max_capacity}")
        print(
            f"      Injection destinations: {requirements['injection_destinations_count']} (includes self: {requirements['includes_self_destination']})")
        print(f"      Child facilities: {requirements['child_facilities_count']} (already counted in injection)")

        # DETAILED MINIMUM CALCULATION DEBUG
        print(f"      --- Sort Points Analysis ---")
        print(f"      Own region destinations: {requirements['own_region_count']} (children + self if hybrid)")
        print(
            f"      External regions served: {requirements['external_regions_count']} (unique regional_sort_hubs - 1)")
        print(f"      Total unique regional sort hubs: {requirements['unique_regional_sort_hubs_count']}")
        print(f"      MINIMUM sort points needed: {requirements['minimum_sort_points']} (optimal region-level)")
        print(f"      MARKET LEVEL sort points needed: {requirements['market_level_sort_points']} (all market-level)")
        print(f"      NOTE: Sort group level would require 4-8× market level due to sort group multipliers")

        # Capacity analysis
        min_deficit = max_capacity - requirements['minimum_sort_points']
        market_deficit = max_capacity - requirements['market_level_sort_points']

        print(
            f"      Capacity vs MINIMUM: {min_deficit} ({'FEASIBLE' if min_deficit >= 0 else 'INFEASIBLE - will fail'})")
        print(
            f"      Capacity vs MARKET LEVEL: {market_deficit} ({'SURPLUS' if market_deficit >= 0 else 'DEFICIT - forces region sorting'})")

    print(f"    Created sort capacity constraints for {len(facility_sort_points)} facilities")

    # Arc volume variables - track packages per arc
    arc_pkgs = {a_idx: model.NewIntVar(0, 1000000, f"arc_pkgs_{a_idx}") for a_idx in range(len(arc_meta))}

    # Link path selection to arc volumes
    for a_idx in range(len(arc_meta)):
        terms = []
        for i in path_keys:
            if a_idx in path_arcs[i]:
                pkgs = int(round(path_od_data[i]['pkgs_day']))
                terms.append(pkgs * x[i])

        if terms:
            model.Add(arc_pkgs[a_idx] == sum(terms))
        else:
            model.Add(arc_pkgs[a_idx] == 0)

    # Calculate truck requirements per arc
    w_cube = weighted_pkg_cube(package_mix)
    raw_trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])

    if strategy.lower() == "container":
        gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util_container = float(gaylord_row["pack_utilization_container"])
        effective_container_cube = raw_container_cube * pack_util_container
        containers_per_truck = int(gaylord_row["containers_per_truck"])
        effective_truck_cube = containers_per_truck * effective_container_cube
    else:
        pack_util_fluid = float(container_params["pack_utilization_fluid"].iloc[0])
        effective_truck_cube = raw_trailer_cube * pack_util_fluid

    effective_truck_cube_scaled = int(effective_truck_cube * 1000)
    w_cube_scaled = int(w_cube * 1000)

    # Arc truck variables
    arc_trucks = {a_idx: model.NewIntVar(0, 1000, f"arc_trucks_{a_idx}") for a_idx in range(len(arc_meta))}

    for a_idx in range(len(arc_meta)):
        model.Add(arc_trucks[a_idx] * effective_truck_cube_scaled >= arc_pkgs[a_idx] * w_cube_scaled)

        arc_has_pkgs = model.NewBoolVar(f"arc_has_pkgs_{a_idx}")
        BIG_M = 1000000
        model.Add(arc_pkgs[a_idx] <= BIG_M * arc_has_pkgs)
        model.Add(arc_pkgs[a_idx] >= 1 * arc_has_pkgs)
        model.Add(arc_trucks[a_idx] >= arc_has_pkgs)

    # Objective with sort-level dependent processing costs
    cost_terms = []

    # 1. Transportation costs
    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        truck_cost = int(arc["cost_per_truck"])
        cost_terms.append(arc_trucks[a_idx] * truck_cost)

    # 2. Sort-level dependent processing costs
    for group_name, group_idxs in groups.items():
        scenario_id, origin, dest, day_type = group_name

        # Get representative volume
        repr_idx = group_idxs[0]
        volume = path_od_data[repr_idx]['pkgs_day']

        # Create auxiliary variables for cost calculation
        sort_vars = sort_decision[group_name]

        # Calculate costs for each sort level
        for sort_level in sort_levels:
            sort_var = sort_vars.get(sort_level)
            if sort_var is None:
                continue

            # Calculate processing cost for this sort level
            repr_od_row = pd.Series(path_od_data[repr_idx])
            costs = calculate_sort_level_costs(repr_od_row, sort_level, cost_kv, facilities)
            total_processing_cost = int(sum(costs.values()))

            # For each path in this group, add cost if both path is selected AND this sort level is chosen
            for path_idx in group_idxs:
                # Create auxiliary variable for path_selected AND sort_level_chosen
                cost_active = model.NewBoolVar(f"cost_active_{path_idx}_{sort_level}")
                model.Add(cost_active <= x[path_idx])
                model.Add(cost_active <= sort_var)
                model.Add(cost_active >= x[path_idx] + sort_var - 1)

                # Add cost term
                cost_terms.append(cost_active * total_processing_cost)

    model.Minimize(sum(cost_terms))

    print(f"    Objective includes {len([t for t in cost_terms if 'arc_trucks' in str(t)])} transportation terms")
    print(
        f"    Plus {len(cost_terms) - len([t for t in cost_terms if 'arc_trucks' in str(t)])} sort-dependent processing terms")

    # Solve with reasonable time limit
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600.0
    solver.parameters.num_search_workers = 8

    print(f"    Starting MILP solver with {len(cost_terms)} cost terms...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"    ❌ MILP solver failed with status: {status}")
        if status == cp_model.INFEASIBLE:
            print("    💡 This likely means sort capacity constraints are too restrictive.")
            print("    💡 Check max_sort_points_capacity values in your facilities sheet.")
            print("    💡 Consider increasing capacity values or reviewing sort_points_per_destination parameter.")
        return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame()

    print(f"    ✅ MILP solver completed with status: {status}")

    # Extract solution
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]
    total_cost_unscaled = solver.ObjectiveValue()
    print(f"    Total optimized cost: ${total_cost_unscaled:,.0f}")

    # Extract sort level decisions
    sort_decisions = {}
    for group_name in od_groups:
        for sort_level in sort_levels:
            sort_var = sort_decision[group_name].get(sort_level)
            if sort_var is not None and solver.Value(sort_var) == 1:
                sort_decisions[group_name] = sort_level
                break

    print(f"    Sort level decisions: {len(sort_decisions)} OD groups optimized")

    # Build selected paths dataframe with sort level info
    selected_paths_data = []

    for i in chosen_idx:
        path_data = path_od_data[i]

        # Get sort level decision for this path
        group_key = (path_data['scenario_id'], path_data['origin'], path_data['dest'], path_data['day_type'])
        chosen_sort_level = sort_decisions.get(group_key, 'market')

        # Calculate costs for chosen sort level
        od_row = pd.Series(path_data)
        sort_costs = calculate_sort_level_costs(od_row, chosen_sort_level, cost_kv, facilities)
        total_processing_cost = sum(sort_costs.values())

        # Calculate transportation cost allocation
        total_transport_cost = 0
        for a_idx in path_arcs[i]:
            arc = arc_meta[a_idx]
            trucks_on_arc = solver.Value(arc_trucks[a_idx])
            if trucks_on_arc > 0:
                arc_total_pkgs = solver.Value(arc_pkgs[a_idx])
                path_share = path_data['pkgs_day'] / max(arc_total_pkgs, 1e-9)
                allocated_transport_cost = trucks_on_arc * arc['cost_per_truck'] * path_share
                total_transport_cost += allocated_transport_cost

        total_path_cost = total_transport_cost + total_processing_cost

        selected_paths_data.append({
            **path_data,
            'total_cost': total_path_cost,
            'linehaul_cost': total_transport_cost,
            'processing_cost': total_processing_cost,
            'cost_per_pkg': total_path_cost / path_data['pkgs_day'],
            'chosen_sort_level': chosen_sort_level,
            'injection_sort_cost': sort_costs['injection_sort_cost'],
            'intermediate_crossdock_cost': sort_costs['intermediate_crossdock_cost'],
            'parent_hub_sort_cost': sort_costs['parent_hub_sort_cost'],
            'last_mile_sort_cost': sort_costs['last_mile_sort_cost'],
            'last_mile_delivery_cost': sort_costs['last_mile_delivery_cost']
        })

    selected_paths = pd.DataFrame(selected_paths_data)

    # Build arc summary
    arc_summary_data = []
    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        pkgs = solver.Value(arc_pkgs[a_idx])
        trucks = solver.Value(arc_trucks[a_idx])

        if pkgs > 0:
            total_cost = trucks * arc['cost_per_truck']
            cube = pkgs * w_cube

            if strategy.lower() == "container":
                gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
                raw_container_cube = float(gaylord_row["usable_cube_cuft"])
                pack_util_container = float(gaylord_row["pack_utilization_container"])
                effective_container_cube = raw_container_cube * pack_util_container
                containers_per_truck = int(gaylord_row["containers_per_truck"])

                actual_containers = max(1, int(np.ceil(cube / effective_container_cube)))
                container_fill_rate = cube / (actual_containers * raw_container_cube)
                truck_fill_rate = cube / (trucks * raw_trailer_cube)
            else:
                container_fill_rate = 0.0
                actual_containers = 0
                truck_fill_rate = cube / (trucks * raw_trailer_cube)

            max_effective_capacity = trucks * effective_truck_cube
            if cube > max_effective_capacity:
                excess_cube = cube - max_effective_capacity
                packages_dwelled = excess_cube / w_cube
            else:
                packages_dwelled = 0
            packages_dwelled = max(0, packages_dwelled)

            scenario_id = "default"
            day_type = "peak"
            for i in chosen_idx:
                if a_idx in path_arcs[i]:
                    scenario_id = path_od_data[i]['scenario_id']
                    day_type = path_od_data[i]['day_type']
                    break

            arc_summary_data.append({
                "scenario_id": scenario_id,
                "day_type": day_type,
                "from_facility": arc["from"],
                "to_facility": arc["to"],
                "distance_miles": arc["distance_miles"],
                "pkgs_day": pkgs,
                "pkg_cube_cuft": cube,
                "trucks": trucks,
                "physical_containers": actual_containers,
                "packages_per_truck": pkgs / trucks,
                "cube_per_truck": cube / trucks,
                "container_fill_rate": container_fill_rate,
                "truck_fill_rate": truck_fill_rate,
                "packages_dwelled": packages_dwelled,
                "cost_per_truck": arc["cost_per_truck"],
                "total_cost": total_cost,
                "CPP": total_cost / pkgs,
            })

    arc_summary = pd.DataFrame(arc_summary_data).sort_values(
        ["scenario_id", "day_type", "from_facility", "to_facility"]
    ).reset_index(drop=True)

    print(f"    Selected {len(selected_paths)} optimal paths using {len(arc_summary)} arcs")

    # Calculate network-level KPIs
    network_kpis = {}
    if not arc_summary.empty:
        total_cube_used = arc_summary['pkg_cube_cuft'].sum()
        total_cube_capacity = (arc_summary['trucks'] * raw_trailer_cube).sum()
        network_truck_fill = total_cube_used / total_cube_capacity if total_cube_capacity > 0 else 0

        total_volume = arc_summary['pkgs_day'].sum()
        if total_volume > 0:
            network_container_fill = (arc_summary['container_fill_rate'] * arc_summary['pkgs_day']).sum() / total_volume
        else:
            network_container_fill = 0.0

        total_dwelled = arc_summary['packages_dwelled'].sum()

        network_kpis = {
            "avg_truck_fill_rate": max(0.0, min(1.0, network_truck_fill)),
            "avg_container_fill_rate": max(0.0, min(1.0, network_container_fill)),
            "total_packages_dwelled": max(0, total_dwelled)
        }
    else:
        network_kpis = {
            "avg_truck_fill_rate": 0.0,
            "avg_container_fill_rate": 0.0,
            "total_packages_dwelled": 0
        }

    # Build sort summary with baseline comparison
    from .sort_optimization import build_sort_decision_summary

    # Convert sort decisions to format expected by summary function
    od_sort_decisions = {}
    for group_name, sort_level in sort_decisions.items():
        scenario_id, origin, dest, day_type = group_name
        od_sort_decisions[(origin, dest)] = sort_level

    sort_summary = build_sort_decision_summary(selected_paths, od_sort_decisions, cost_kv, facilities)

    # Calculate baseline metrics (all market-level sort)
    print(f"    Calculating baseline metrics (unconstrained market-level sort)...")
    baseline_total_cost = 0
    baseline_processing_cost = 0
    baseline_transport_cost = 0

    for i in chosen_idx:
        path_data = path_od_data[i]

        # Calculate baseline costs (market level for all)
        od_row = pd.Series(path_data)
        baseline_sort_costs = calculate_sort_level_costs(od_row, 'market', cost_kv, facilities)
        baseline_path_processing = sum(baseline_sort_costs.values())
        baseline_processing_cost += baseline_path_processing

        # Transportation cost remains the same
        baseline_transport_cost = total_transport_cost = 0
        for a_idx in path_arcs[i]:
            arc = arc_meta[a_idx]
            trucks_on_arc = solver.Value(arc_trucks[a_idx])
            if trucks_on_arc > 0:
                arc_total_pkgs = solver.Value(arc_pkgs[a_idx])
                path_share = path_data['pkgs_day'] / max(arc_total_pkgs, 1e-9)
                allocated_transport_cost = trucks_on_arc * arc['cost_per_truck'] * path_share
                baseline_transport_cost += allocated_transport_cost

        baseline_total_cost += baseline_path_processing + baseline_transport_cost

    # Calculate total packages for per-package metrics
    total_packages = sum(path_od_data[i]['pkgs_day'] for i in chosen_idx)
    baseline_cost_per_pkg = baseline_total_cost / max(total_packages, 1)

    # Current optimized costs
    optimized_total_cost = selected_paths["total_cost"].sum()
    optimized_cost_per_pkg = optimized_total_cost / max(total_packages, 1)

    # Calculate constraint impact
    constraint_cost_impact = optimized_total_cost - baseline_total_cost
    constraint_cost_impact_pct = (constraint_cost_impact / baseline_total_cost) * 100 if baseline_total_cost > 0 else 0

    print(f"    Baseline (market-level): ${baseline_total_cost:,.0f} (${baseline_cost_per_pkg:.3f}/pkg)")
    print(f"    Optimized (constrained): ${optimized_total_cost:,.0f} (${optimized_cost_per_pkg:.3f}/pkg)")
    print(f"    Capacity constraint impact: ${constraint_cost_impact:,.0f} ({constraint_cost_impact_pct:+.1f}%)")

    # Add sort optimization metrics to network KPIs
    if not sort_summary.empty:
        sort_level_counts = sort_summary['chosen_sort_level'].value_counts()
        total_ods = len(sort_summary)

        network_kpis.update({
            'pct_region_sort': (sort_level_counts.get('region', 0) / total_ods),
            'pct_market_sort': (sort_level_counts.get('market', 0) / total_ods),
            'pct_sort_group_sort': (sort_level_counts.get('sort_group', 0) / total_ods),
            'total_sort_savings': sort_summary['savings_vs_market'].sum(),
            'baseline_total_cost': baseline_total_cost,
            'baseline_cost_per_pkg': baseline_cost_per_pkg,
            'baseline_processing_cost': baseline_processing_cost,
            'optimized_total_cost': optimized_total_cost,
            'optimized_cost_per_pkg': optimized_cost_per_pkg,
            'constraint_cost_impact': constraint_cost_impact,
            'constraint_cost_impact_pct': constraint_cost_impact_pct / 100,  # Convert to decimal
        })

    print(
        f"    Sort level distribution: Region {network_kpis.get('pct_region_sort', 0):.1f}%, Market {network_kpis.get('pct_market_sort', 0):.1f}%, Sort Group {network_kpis.get('pct_sort_group_sort', 0):.1f}%")

    return selected_paths, arc_summary, network_kpis, sort_summary


# Keep original function for backward compatibility
def solve_arc_pooled_path_selection(
        candidates: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        cost_kv: Dict[str, float],
        timing_kv: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Original MILP solver - now calls extended version and returns compatible results.
    """
    selected_paths, arc_summary, network_kpis, _ = solve_arc_pooled_path_selection_with_sort_optimization(
        candidates, facilities, mileage_bands, package_mix, container_params, cost_kv, timing_kv
    )

    return selected_paths, arc_summary, network_kpis