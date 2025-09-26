# veho_net/milp.py - EXTENDED: Multi-level sortation optimization integrated with path selection
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from .time_cost import weighted_pkg_cube, calculate_truck_capacity
from .geo import haversine_miles, band_lookup
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
    EXTENDED ARCHITECTURE: Path selection with integrated multi-level sort optimization.

    Returns:
        - od_selected: Optimal paths with sort level decisions
        - arc_summary: Arc/lane utilization
        - network_kpis: Network performance metrics
        - sort_summary: Sort level decision analysis
    """
    cand = candidates.reset_index(drop=True).copy()
    strategy = str(cost_kv.get("load_strategy", "container")).lower()

    print(f"    EXTENDED MILP: Path selection with multi-level sort optimization")
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

    # EXTENDED: Sort level decision variables for each OD group
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

    # EXTENDED: Sort point capacity constraints - CORRECTED to handle aggregate workload
    facility_sort_points = {}

    # DEBUG: Count unique parent hubs (regions) for capacity validation
    unique_parent_hubs = set()
    for facility_name in facilities['facility_name']:
        if facility_name in facility_lookup.index:
            parent_hub = relationships['parent_hub_map'][facility_name]
            unique_parent_hubs.add(parent_hub)

    print(f"    DEBUG: Found {len(unique_parent_hubs)} unique regions (parent hubs)")
    print(f"    DEBUG: sort_points_per_dest = {sort_points_per_dest}")

    # Create capacity constraints for ALL facilities that could consume sort points
    all_relevant_facilities = set(hub_facilities)

    # Add parent hubs to the list (they need capacity for region-level breakdown)
    for facility_name in facilities['facility_name']:
        if facility_name in facility_lookup.index:
            parent_hub = relationships['parent_hub_map'][facility_name]
            if parent_hub in facility_lookup.index and facility_lookup.at[parent_hub, 'type'] in ['hub', 'hybrid']:
                all_relevant_facilities.add(parent_hub)

    for facility_name in all_relevant_facilities:
        if facility_name not in facility_lookup.index:
            continue

        # Get max capacity - fail if missing for facilities that need it
        max_capacity = facility_lookup.at[facility_name, 'max_sort_points_capacity']

        if pd.isna(max_capacity) or max_capacity <= 0:
            # Only require capacity for facilities that actually process sort points
            facility_role = "injection" if facility_name in hub_facilities else "parent_hub"
            print(f"    WARNING: {facility_name} ({facility_role}) missing capacity - setting to 1000 for now")
            max_capacity = 1000
        else:
            max_capacity = int(max_capacity)

        # Create sort point usage variable
        facility_sort_points[facility_name] = model.NewIntVar(0, max_capacity, f"sort_points_{facility_name}")

        sort_point_terms = []

        # INJECTION FACILITY ROLE: Handles market and sort group sorting
        if facility_name in hub_facilities:
            destinations_from_facility = set()
            for group_name, group_idxs in groups.items():
                scenario_id, origin, dest, day_type = group_name
                if origin == facility_name:
                    destinations_from_facility.add(dest)

            if destinations_from_facility:
                print(f"    {facility_name} (injection): serves {len(destinations_from_facility)} destinations")

                # Create destination-level sort decision variables for this injection facility
                dest_sort_decisions = {}

                for dest in destinations_from_facility:
                    dest_sort_decisions[dest] = {}

                    # Get sample OD to determine valid sort levels
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
                    if origin == facility_name:
                        od_sort_vars = sort_decision[group_name]
                        dest_sort_vars = dest_sort_decisions[dest]

                        for sort_level in ['region', 'market', 'sort_group']:
                            if od_sort_vars[sort_level] is not None and dest_sort_vars[sort_level] is not None:
                                model.Add(od_sort_vars[sort_level] == dest_sort_vars[sort_level])

                # INJECTION FACILITY: Only market and sort group consume sort points here
                # BUT: If this injection facility is ALSO the parent hub for a destination,
                # then region-level should NOT consume sort points here (it's handled internally)

                # Market level: sort points at injection facility
                for dest in destinations_from_facility:
                    if dest_sort_decisions[dest]['market'] is not None:
                        sort_point_terms.append(dest_sort_decisions[dest]['market'] * int(sort_points_per_dest))

                # Sort group level: sort points at injection facility
                for dest in destinations_from_facility:
                    if dest_sort_decisions[dest]['sort_group'] is not None:
                        if dest in facility_lookup.index:
                            sort_groups_count = facility_lookup.at[dest, 'last_mile_sort_groups_count']
                            if not pd.isna(sort_groups_count):
                                sort_group_points = int(sort_points_per_dest * sort_groups_count)
                                sort_point_terms.append(dest_sort_decisions[dest]['sort_group'] * sort_group_points)

                # Region level: ONLY consume sort points if destination's parent hub is DIFFERENT
                # (If same parent hub, it's handled internally without consuming sort points)
                destinations_needing_external_region_sort = set()
                for dest in destinations_from_facility:
                    dest_parent_hub = relationships['parent_hub_map'][dest]
                    if dest_parent_hub != facility_name:  # External parent hub
                        destinations_needing_external_region_sort.add(dest_parent_hub)

                if destinations_needing_external_region_sort:
                    print(
                        f"      External region sorting to {len(destinations_needing_external_region_sort)} parent hubs")

                    # Create sort points for each external parent hub region
                    for external_parent_hub in destinations_needing_external_region_sort:
                        # Check if any destinations going to this external parent hub use region sorting
                        external_region_active = model.NewBoolVar(
                            f"external_region_{facility_name}_{external_parent_hub}")

                        region_terms = []
                        for dest in destinations_from_facility:
                            dest_parent_hub = relationships['parent_hub_map'][dest]
                            if dest_parent_hub == external_parent_hub:
                                if dest_sort_decisions[dest]['region'] is not None:
                                    region_terms.append(dest_sort_decisions[dest]['region'])

                        if region_terms:
                            model.Add(external_region_active <= sum(region_terms))
                            for term in region_terms:
                                model.Add(external_region_active >= term)

                            # Add sort points for external region sorting
                            sort_point_terms.append(external_region_active * int(sort_points_per_dest))

        # PARENT HUB ROLE: Aggregate sorting workload from all sources
        # Parent hubs handle multiple types of sorting operations:
        # 1. Own injection volume sorting to other facilities
        # 2. Region container breakdown from other injection facilities
        # 3. Child facility processing (always required regardless of inbound sort level)

        # First, check if this facility is a parent hub with child facilities
        child_facilities = []
        for child_name in facilities['facility_name']:
            if child_name in facility_lookup.index:
                child_parent_hub = relationships['parent_hub_map'][child_name]
                if child_parent_hub == facility_name and child_name != facility_name:
                    child_facilities.append(child_name)

        # Check if this facility receives region containers from other injection facilities
        external_region_breakdown_destinations = set()
        for group_name, group_idxs in groups.items():
            scenario_id, origin, dest, day_type = group_name
            dest_parent_hub = relationships['parent_hub_map'][dest]
            # If this facility is parent hub AND origin is different injection facility
            # AND destination is NOT a child facility (to avoid double counting)
            if dest_parent_hub == facility_name and origin != facility_name and dest not in child_facilities:
                od_sort_vars = sort_decision[group_name]
                if od_sort_vars['region'] is not None:
                    external_region_breakdown_destinations.add(dest)

        if child_facilities or destinations_from_facility:
            # Calculate capacity requirements and availability for this facility
            # (handles both parent hub and injection roles in one summary)

            child_count = len(child_facilities)

            # For injection role: destinations this facility injects to
            injection_destinations = len(destinations_from_facility) if facility_name in hub_facilities else 0

            # Calculate sort point requirements
            sort_points_for_children = child_count * int(sort_points_per_dest)
            sort_points_for_injection_markets = injection_destinations * int(sort_points_per_dest)
            total_sort_points_needed = sort_points_for_children + sort_points_for_injection_markets

            # Sort point deficit/surplus analysis (positive = surplus, negative = deficit)
            sort_point_deficit = max_capacity - total_sort_points_needed

            # Build role description
            roles = []
            if child_count > 0:
                roles.append(f"parent_hub({child_count} children)")
            if injection_destinations > 0:
                roles.append(f"injection({injection_destinations} destinations)")
            role_desc = " + ".join(roles)

            print(f"    {facility_name} ({role_desc}): {sort_points_per_dest} sort_points/dest")
            print(f"      Available capacity: {max_capacity}")
            if child_count > 0:
                print(f"      Children need: {sort_points_for_children}")
            if injection_destinations > 0:
                print(f"      Injection markets need: {sort_points_for_injection_markets}")
            print(f"      Total needed: {total_sort_points_needed}")
            print(
                f"      Sort point deficit: {sort_point_deficit} ({'SURPLUS - allows sort group opportunities' if sort_point_deficit > 0 else 'DEFICIT - will force region level rollup'})")

            # OPERATION 1: Child facility sorting (always required)
            # Parent hub must handle all volume destined for child facilities
            for child_facility in child_facilities:
                child_sort_active = model.NewBoolVar(f"child_sort_{facility_name}_{child_facility}")

                # Child facility needs parent hub sorting if it receives any volume
                child_receives_volume = []

                # Check all ODs destined for this child facility
                for group_name, group_idxs in groups.items():
                    scenario_id, origin, dest, day_type = group_name
                    if dest == child_facility:
                        # Path selected to this child
                        path_selected_terms = [x[idx] for idx in group_idxs]
                        path_selected = model.NewBoolVar(f"path_selected_{scenario_id}_{origin}_{dest}_{day_type}")
                        model.Add(path_selected == sum(path_selected_terms))

                        # Parent hub sorting is needed UNLESS origin chooses sort_group level
                        od_sort_vars = sort_decision[group_name]

                        # If sort_group is chosen, parent hub doesn't need to sort (crossdock only)
                        # Otherwise, parent hub needs to sort regardless of region/market choice
                        if od_sort_vars['sort_group'] is not None:
                            # Parent hub sorts only if NOT sort_group
                            not_sort_group = model.NewBoolVar(
                                f"not_sort_group_{scenario_id}_{origin}_{dest}_{day_type}")
                            model.Add(not_sort_group + od_sort_vars['sort_group'] == 1)

                            parent_needs_sort = model.NewBoolVar(
                                f"parent_needs_sort_{scenario_id}_{origin}_{dest}_{day_type}")
                            model.Add(parent_needs_sort <= path_selected)
                            model.Add(parent_needs_sort <= not_sort_group)
                            model.Add(parent_needs_sort >= path_selected + not_sort_group - 1)
                            child_receives_volume.append(parent_needs_sort)
                        else:
                            # No sort_group option available, parent always sorts
                            child_receives_volume.append(path_selected)

                if child_receives_volume:
                    # Child sort is active if any volume needs parent hub sorting
                    model.Add(child_sort_active <= sum(child_receives_volume))
                    for term in child_receives_volume:
                        model.Add(child_sort_active >= term)

                    # Add sort points for child facility processing
                    sort_point_terms.append(child_sort_active * int(sort_points_per_dest))

            # OPERATION 2: Region container breakdown from external injection facilities
            for dest in external_region_breakdown_destinations:
                # Skip if this destination is a child facility (already counted above)
                if dest in child_facilities:
                    continue

                # Create variable for region breakdown to this destination
                region_breakdown_active = model.NewBoolVar(f"region_breakdown_{facility_name}_{dest}")

                # Link to external ODs that send region-level containers here
                external_region_terms = []
                for group_name, group_idxs in groups.items():
                    scenario_id, origin, dest_check, day_type = group_name
                    if dest_check == dest and origin != facility_name:
                        dest_parent_hub_check = relationships['parent_hub_map'][dest]
                        if dest_parent_hub_check == facility_name:
                            od_sort_vars = sort_decision[group_name]
                            if od_sort_vars['region'] is not None:
                                external_region_terms.append(od_sort_vars['region'])

                # Region breakdown is active if any external facility sends region containers
                if external_region_terms:
                    model.Add(region_breakdown_active <= sum(external_region_terms))
                    for term in external_region_terms:
                        model.Add(region_breakdown_active >= term)

                    # Add sort points for external region breakdown
                    sort_point_terms.append(region_breakdown_active * int(sort_points_per_dest))

        # Add capacity constraint
        if sort_point_terms:
            model.Add(facility_sort_points[facility_name] >= sum(sort_point_terms))
        else:
            model.Add(facility_sort_points[facility_name] == 0)

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

    # EXTENDED: Objective with sort-level dependent processing costs
    cost_terms = []

    # 1. Transportation costs (unchanged)
    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        truck_cost = int(arc["cost_per_truck"])
        cost_terms.append(arc_trucks[a_idx] * truck_cost)

    # 2. EXTENDED: Sort-level dependent processing costs
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

    # EXTENDED: Extract sort level decisions
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

    # Build arc summary (unchanged logic)
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

    # EXTENDED: Build sort summary with baseline comparison
    from .sort_optimization import build_sort_decision_summary

    # Convert sort decisions to format expected by summary function
    od_sort_decisions = {}
    for group_name, sort_level in sort_decisions.items():
        scenario_id, origin, dest, day_type = group_name
        od_sort_decisions[(origin, dest)] = sort_level

    sort_summary = build_sort_decision_summary(selected_paths, od_sort_decisions, cost_kv, facilities)

    # EXTENDED: Calculate baseline metrics (all market-level sort)
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
            'pct_region_sort': (sort_level_counts.get('region', 0) / total_ods) * 100,
            'pct_market_sort': (sort_level_counts.get('market', 0) / total_ods) * 100,
            'pct_sort_group_sort': (sort_level_counts.get('sort_group', 0) / total_ods) * 100,
            'total_sort_savings': sort_summary['savings_vs_market'].sum(),
            # EXTENDED: Add baseline comparison metrics
            'baseline_total_cost': baseline_total_cost,
            'baseline_cost_per_pkg': baseline_cost_per_pkg,
            'baseline_processing_cost': baseline_processing_cost,
            'optimized_total_cost': optimized_total_cost,
            'optimized_cost_per_pkg': optimized_cost_per_pkg,
            'constraint_cost_impact': constraint_cost_impact,
            'constraint_cost_impact_pct': constraint_cost_impact_pct,
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