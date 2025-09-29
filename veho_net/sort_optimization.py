# veho_net/sort_optimization.py - FIXED: O=D handling and parent hub sort costs
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def build_facility_relationships(facilities: pd.DataFrame) -> Dict[str, Dict]:
    """
    Build facility relationship mapping with separate routing vs. sorting hierarchies.

    Returns:
        Dict with facility relationships and classification info
    """
    relationships = {}

    # Build parent hub mapping (for routing/paths)
    parent_hub_map = {}
    # Build regional sort hub mapping (for sorting decisions)
    regional_sort_hub_map = {}
    facility_types = {}
    launch_facilities = set()
    hub_facilities = set()

    # Check if regional_sort_hub column exists, fallback to parent_hub_name
    has_regional_sort_hub = 'regional_sort_hub' in facilities.columns

    for _, row in facilities.iterrows():
        facility_name = row['facility_name']
        facility_type = str(row['type']).lower()
        parent_hub = row.get('parent_hub_name', facility_name)

        # Use regional_sort_hub if available, otherwise fallback to parent_hub_name
        if has_regional_sort_hub:
            regional_sort_hub = row.get('regional_sort_hub', parent_hub)
            if pd.isna(regional_sort_hub) or regional_sort_hub == "":
                regional_sort_hub = parent_hub
        else:
            regional_sort_hub = parent_hub

        if pd.isna(parent_hub) or parent_hub == "":
            parent_hub = facility_name
        if pd.isna(regional_sort_hub) or regional_sort_hub == "":
            regional_sort_hub = facility_name

        parent_hub_map[facility_name] = parent_hub
        regional_sort_hub_map[facility_name] = regional_sort_hub
        facility_types[facility_name] = facility_type

        if facility_type == 'launch':
            launch_facilities.add(facility_name)
        elif facility_type in ['hub', 'hybrid']:
            hub_facilities.add(facility_name)

    relationships = {
        'parent_hub_map': parent_hub_map,  # Used for routing/paths
        'regional_sort_hub_map': regional_sort_hub_map,  # Used for sorting
        'facility_types': facility_types,
        'launch_facilities': launch_facilities,
        'hub_facilities': hub_facilities,
        'has_regional_sort_hub': has_regional_sort_hub
    }

    return relationships


def calculate_sort_point_requirements(od_data: pd.DataFrame, facilities: pd.DataFrame,
                                      timing_kv: Dict, sort_level: str) -> Dict[str, Dict]:
    """
    Calculate sort point requirements for each facility based on sort level choice.
    Uses regional_sort_hub for sorting hierarchy.

    Args:
        od_data: OD pairs with volume
        facilities: Facility data with capacity info
        timing_kv: Timing parameters including sort_points_per_destination
        sort_level: 'region', 'market', or 'sort_group'

    Returns:
        Dict[facility_name, Dict[destination_key, sort_points_needed]]
    """
    relationships = build_facility_relationships(facilities)
    regional_sort_hub_map = relationships['regional_sort_hub_map']

    sort_points_per_dest = float(timing_kv.get('sort_points_per_destination', 1.0))
    facility_requirements = defaultdict(lambda: defaultdict(float))

    # Create facility lookup for sort groups count
    facility_lookup = facilities.set_index('facility_name')

    for _, od_row in od_data.iterrows():
        origin = od_row['origin']
        dest = od_row['dest']
        volume = od_row.get('pkgs_day', 0)

        if volume <= 0:
            continue

        # Determine destination key and sort points based on sort level
        if sort_level == 'region':
            # Sort to destination regional sort hub (region)
            dest_key = regional_sort_hub_map[dest]
            sort_points = sort_points_per_dest

        elif sort_level == 'market':
            # Sort to destination facility (market)
            dest_key = dest
            sort_points = sort_points_per_dest

        elif sort_level == 'sort_group':
            # Sort to destination facility sort groups
            dest_key = dest
            sort_groups_count = facility_lookup.at[
                dest, 'last_mile_sort_groups_count'] if dest in facility_lookup.index else 4
            sort_points = sort_points_per_dest * float(sort_groups_count)
        else:
            raise ValueError(f"Invalid sort level: {sort_level}")

        # Add sort point requirement at origin facility
        facility_requirements[origin][dest_key] += sort_points

    # Convert to regular dict for easier handling
    return {facility: dict(dest_reqs) for facility, dest_reqs in facility_requirements.items()}


def aggregate_facility_volumes(od_selected: pd.DataFrame, facilities: pd.DataFrame) -> Dict[str, Dict]:
    """
    Aggregate volume flows by facility role (injection, intermediate, destination).
    Uses regional_sort_hub for regional aggregation.

    Returns:
        Dict[facility_name, {injection_volume, intermediate_volume, destination_volume}]
    """
    relationships = build_facility_relationships(facilities)
    regional_sort_hub_map = relationships['regional_sort_hub_map']

    facility_volumes = defaultdict(lambda: {
        'injection_volume': 0.0,
        'intermediate_volume': 0.0,
        'destination_volume': 0.0,
        'injection_ods': [],
        'intermediate_ods': [],
        'destination_ods': []
    })

    for _, od_row in od_selected.iterrows():
        origin = od_row['origin']
        dest = od_row['dest']
        volume = float(od_row.get('pkgs_day', 0))
        path_str = str(od_row.get('path_str', f"{origin}->{dest}"))

        if volume <= 0:
            continue

        # Parse path nodes
        if '->' in path_str:
            nodes = [n.strip() for n in path_str.split('->')]
        else:
            nodes = [origin, dest]

        # Track volumes by facility role
        for i, node in enumerate(nodes):
            if i == 0:
                # Origin = injection
                facility_volumes[node]['injection_volume'] += volume
                facility_volumes[node]['injection_ods'].append((origin, dest))
            elif i == len(nodes) - 1:
                # Destination
                facility_volumes[node]['destination_volume'] += volume
                facility_volumes[node]['destination_ods'].append((origin, dest))
            else:
                # Intermediate
                facility_volumes[node]['intermediate_volume'] += volume
                facility_volumes[node]['intermediate_ods'].append((origin, dest))

    return dict(facility_volumes)


def calculate_sort_level_costs(od_row: pd.Series, sort_level: str, cost_kv: Dict,
                               facilities: pd.DataFrame) -> Dict[str, float]:
    """
    FIXED: Calculate processing costs for an OD pair based on chosen sort level.
    Handles O=D correctly (no linehaul, no intermediate costs).
    Uses regional_sort_hub for regional cost calculations.

    Returns:
        Dict with cost breakdown by component
    """
    relationships = build_facility_relationships(facilities)
    regional_sort_hub_map = relationships['regional_sort_hub_map']
    facility_types = relationships['facility_types']

    # Get cost parameters
    injection_sort_pp = float(cost_kv.get("injection_sort_cost_per_pkg",
                                          cost_kv.get("sort_cost_per_pkg", 0.0)))
    intermediate_crossdock_pp = float(cost_kv.get("intermediate_sort_cost_per_pkg",
                                                  cost_kv.get("sort_cost_per_pkg", 0.0)))
    parent_hub_sort_pp = float(cost_kv.get("parent_hub_sort_cost_per_pkg",
                                           cost_kv.get("sort_cost_per_pkg", 0.0)))
    last_mile_sort_pp = float(cost_kv.get("last_mile_sort_cost_per_pkg", 0.0))
    last_mile_delivery_pp = float(cost_kv.get("last_mile_delivery_cost_per_pkg", 0.0))

    volume = float(od_row.get('pkgs_day', 0))
    origin = od_row['origin']
    dest = od_row['dest']
    path_str = str(od_row.get('path_str', f"{origin}->{dest}"))

    # Parse path to identify intermediate facilities
    if '->' in path_str:
        nodes = [n.strip() for n in path_str.split('->')]
    else:
        nodes = [origin, dest]

    # FIXED: Check for O=D (self-destination)
    is_self_destination = (origin == dest)

    # Initialize costs
    costs = {
        'injection_sort_cost': injection_sort_pp * volume,
        'intermediate_crossdock_cost': 0.0,
        'parent_hub_sort_cost': 0.0,
        'last_mile_sort_cost': 0.0,
        'last_mile_delivery_cost': last_mile_delivery_pp * volume
    }

    # FIXED: For O=D, only apply injection sort + last mile delivery
    # No intermediate costs, no parent hub sort, and sort level is always sort_group
    if is_self_destination:
        # O=D must use sort_group level (facility sorting for its own delivery)
        costs['last_mile_sort_cost'] = 0.0  # Already sorted at injection
        costs['intermediate_crossdock_cost'] = 0.0  # No crossdock
        costs['parent_hub_sort_cost'] = 0.0  # No parent hub
        return costs

    # For O≠D, process normally
    intermediate_facilities = nodes[1:-1] if len(nodes) > 2 else []
    dest_regional_sort_hub = regional_sort_hub_map[dest]

    # Intermediate crossdock costs (always apply for O≠D with intermediate stops)
    costs['intermediate_crossdock_cost'] = len(intermediate_facilities) * intermediate_crossdock_pp * volume

    # Apply sort level logic for O≠D only
    if sort_level == 'region':
        # Region level: regional sort hub sorts if it's in the path and different from dest
        if dest_regional_sort_hub != dest and dest_regional_sort_hub in nodes:
            costs['parent_hub_sort_cost'] = parent_hub_sort_pp * volume
        # Last mile facility also sorts after regional hub
        costs['last_mile_sort_cost'] = last_mile_sort_pp * volume

    elif sort_level == 'market':
        # Market level: last mile facility sorts (current approach)
        costs['last_mile_sort_cost'] = last_mile_sort_pp * volume

    elif sort_level == 'sort_group':
        # Sort group level: no last mile sort needed (pre-sorted to route groups)
        costs['last_mile_sort_cost'] = 0.0
    else:
        raise ValueError(f"Invalid sort level: {sort_level}")

    return costs


def validate_sort_capacity_constraints(sort_requirements: Dict, facilities: pd.DataFrame) -> Dict:
    """
    Validate that sort point requirements don't exceed facility capacities.

    Returns:
        Dict with validation results and constraint violations
    """
    validation_results = {
        'valid': True,
        'violations': [],
        'utilization': {}
    }

    facility_lookup = facilities.set_index('facility_name')

    for facility_name, dest_requirements in sort_requirements.items():
        if facility_name not in facility_lookup.index:
            continue

        # Get facility capacity
        max_capacity = facility_lookup.at[facility_name, 'max_sort_points_capacity']
        if pd.isna(max_capacity):
            max_capacity = 0
        max_capacity = float(max_capacity)

        # Calculate total required sort points
        total_required = sum(dest_requirements.values())
        utilization = total_required / max(max_capacity, 1e-9)

        validation_results['utilization'][facility_name] = {
            'required': total_required,
            'capacity': max_capacity,
            'utilization_pct': utilization
        }

        # Check for violations
        if total_required > max_capacity:
            validation_results['valid'] = False
            validation_results['violations'].append({
                'facility': facility_name,
                'required': total_required,
                'capacity': max_capacity,
                'excess': total_required - max_capacity
            })

    return validation_results


def get_sort_level_options(od_row: pd.Series, facilities: pd.DataFrame) -> List[str]:
    """
    FIXED: Determine valid sort level options for an OD pair.

    Business Rules:
    - O=D (self-destination): MUST use sort_group (facility sorting for its own delivery)
    - O≠D (middle-mile): Can use region, market, or sort_group

    Returns:
        List of valid sort levels
    """
    origin = od_row['origin']
    dest = od_row['dest']

    # FIXED: O=D must use sort_group level only
    if origin == dest:
        return ['sort_group']

    # O≠D can use all three levels
    return ['region', 'market', 'sort_group']


def calculate_sort_level_savings(od_row: pd.Series, cost_kv: Dict, facilities: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate potential cost savings for each sort level compared to market level baseline.

    Returns:
        Dict[sort_level, cost_savings] (positive = savings, negative = additional cost)
    """
    valid_options = get_sort_level_options(od_row, facilities)

    # Calculate costs for each valid sort level
    level_costs = {}
    for sort_level in valid_options:
        costs = calculate_sort_level_costs(od_row, sort_level, cost_kv, facilities)
        level_costs[sort_level] = sum(costs.values())

    # Use market level as baseline for savings calculation
    baseline_cost = level_costs.get('market', 0)

    savings = {}
    for sort_level, total_cost in level_costs.items():
        savings[sort_level] = baseline_cost - total_cost

    return savings


def build_sort_decision_summary(od_selected: pd.DataFrame, sort_decisions: Dict,
                                cost_kv: Dict, facilities: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED: Build summary of sort level decisions with accurate cost breakdown.

    Returns:
        DataFrame with sort decision analysis
    """
    relationships = build_facility_relationships(facilities)
    regional_sort_hub_map = relationships['regional_sort_hub_map']

    summary_data = []

    for _, od_row in od_selected.iterrows():
        od_key = (od_row['origin'], od_row['dest'])
        chosen_sort_level = sort_decisions.get(od_key, 'market')

        # Calculate costs for chosen level
        costs = calculate_sort_level_costs(od_row, chosen_sort_level, cost_kv, facilities)
        savings = calculate_sort_level_savings(od_row, cost_kv, facilities)

        summary_data.append({
            'origin': od_row['origin'],
            'origin_regional_sort_hub': regional_sort_hub_map.get(od_row['origin'], od_row['origin']),
            'dest': od_row['dest'],
            'dest_regional_sort_hub': regional_sort_hub_map.get(od_row['dest'], od_row['dest']),
            'pkgs_day': od_row.get('pkgs_day', 0),
            'is_self_destination': (od_row['origin'] == od_row['dest']),
            'chosen_sort_level': chosen_sort_level,
            'total_sort_cost': sum(costs.values()),
            'injection_sort_cost': costs['injection_sort_cost'],
            'intermediate_crossdock_cost': costs['intermediate_crossdock_cost'],
            'parent_hub_sort_cost': costs['parent_hub_sort_cost'],
            'last_mile_sort_cost': costs['last_mile_sort_cost'],
            'last_mile_delivery_cost': costs['last_mile_delivery_cost'],
            'savings_vs_market': savings.get(chosen_sort_level, 0),
            'valid_sort_options': ','.join(get_sort_level_options(od_row, facilities))
        })

    return pd.DataFrame(summary_data)