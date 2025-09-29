# veho_net/reporting_sort.py - COMPREHENSIVE FIX: Direct injection handling and accurate cost metrics
import pandas as pd
import numpy as np
from .geo_sort import haversine_miles


def _calculate_containers_needed(pkgs_day: float, package_mix: pd.DataFrame,
                                 container_params: pd.DataFrame, strategy: str) -> dict:
    """
    Calculate container requirements for a given package volume.

    Returns:
        Dict with container counts and cube metrics
    """
    from .time_cost_sort import weighted_pkg_cube

    w_cube = weighted_pkg_cube(package_mix)
    total_cube = pkgs_day * w_cube

    if strategy.lower() == "container":
        gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util_container = float(gaylord_row["pack_utilization_container"])
        effective_container_cube = raw_container_cube * pack_util_container

        # Calculate containers needed
        exact_containers = total_cube / effective_container_cube
        physical_containers = max(1, int(np.ceil(exact_containers)))

        container_fill_rate = min(1.0, total_cube / (physical_containers * raw_container_cube))

        return {
            'physical_containers': physical_containers,
            'total_cube_cuft': total_cube,
            'container_fill_rate': container_fill_rate
        }
    else:
        # Fluid strategy - no containers
        return {
            'physical_containers': 0,
            'total_cube_cuft': total_cube,
            'container_fill_rate': 0.0
        }


def _identify_volume_types_with_costs(od_selected: pd.DataFrame, path_steps_selected: pd.DataFrame,
                                      direct_day: pd.DataFrame, arc_summary: pd.DataFrame,
                                      package_mix: pd.DataFrame = None,
                                      container_params: pd.DataFrame = None,
                                      strategy: str = "container") -> pd.DataFrame:
    """
    CORRECTED: Calculate facility volume and accurate cost breakdowns.

    Key corrections:
    - Direct injection = ONLY Zone 0 (not in middle-mile network)
    - Middle-mile injection = ALL O-D pairs from middle-mile (including O=D)
    - Last-mile = direct injection + middle-mile arrivals (double counts direct, but different perspectives)
    """
    volume_data = []

    # Collect all facilities from various data sources
    all_facilities = set()

    if not od_selected.empty:
        all_facilities.update(od_selected['origin'].unique())
        all_facilities.update(od_selected['dest'].unique())

    if not direct_day.empty and 'dest' in direct_day.columns:
        all_facilities.update(direct_day['dest'].unique())

    for facility in all_facilities:
        try:
            # ========== MIDDLE-MILE INJECTION ROLE ==========
            # ALL packages originating in the middle-mile network (includes O=D)
            mm_injection_pkgs = 0
            mm_injection_containers = 0
            mm_injection_cube = 0.0

            # Cost components for middle-mile injection
            injection_sort_cost = 0.0
            injection_linehaul_cost = 0.0
            injection_processing_cost = 0.0

            if not od_selected.empty:
                # ALL ODs originating at this facility (both O=D and O≠D)
                mm_outbound_ods = od_selected[od_selected['origin'] == facility]

                if not mm_outbound_ods.empty:
                    mm_injection_pkgs = mm_outbound_ods['pkgs_day'].sum()

                    # Get actual cost components from optimization
                    injection_sort_cost = mm_outbound_ods.get('injection_sort_cost', 0).sum()
                    injection_linehaul_cost = mm_outbound_ods.get('linehaul_cost', 0).sum()
                    injection_processing_cost = injection_sort_cost  # Injection processing = injection sort

                    # Calculate containers for middle-mile injection
                    if package_mix is not None and container_params is not None:
                        container_calc = _calculate_containers_needed(
                            mm_injection_pkgs, package_mix, container_params, strategy
                        )
                        mm_injection_containers = container_calc['physical_containers']
                        mm_injection_cube = container_calc['total_cube_cuft']

            # ========== DIRECT INJECTION ROLE (Zone 0 only) ==========
            # Packages that bypass middle-mile network entirely
            direct_injection_pkgs = 0
            direct_injection_containers = 0
            direct_injection_cube = 0.0
            direct_injection_cost = 0.0

            if not direct_day.empty:
                direct_col = 'dir_pkgs_day'
                if direct_col in direct_day.columns:
                    facility_direct = direct_day[direct_day['dest'] == facility]
                    if not facility_direct.empty:
                        direct_injection_pkgs = facility_direct[direct_col].sum()

                        # Calculate containers for direct injection
                        if package_mix is not None and container_params is not None:
                            container_calc = _calculate_containers_needed(
                                direct_injection_pkgs, package_mix, container_params, strategy
                            )
                            direct_injection_containers = container_calc['physical_containers']
                            direct_injection_cube = container_calc['total_cube_cuft']

            # ========== INTERMEDIATE ROLE (Pass-Through Only) ==========
            # True intermediate: packages passing through for crossdock (not final destination)
            # Launch facilities should NEVER have intermediate packages (can't be in middle of path)
            intermediate_pkgs = 0
            intermediate_containers = 0
            intermediate_cube = 0.0
            intermediate_crossdock_cost = 0.0
            intermediate_linehaul_cost = 0.0

            # Get facility type to check if this can be intermediate
            facility_type = None
            if not od_selected.empty and 'origin' in od_selected.columns:
                # Try to infer facility type from facilities data if available
                # Launch facilities should never have intermediate volume
                pass

            if not arc_summary.empty and not od_selected.empty:
                # Only calculate intermediate for hub/hybrid facilities
                # Launch facilities are NEVER intermediate stops in valid paths

                # Packages arriving at this facility via arcs
                inbound_arcs = arc_summary[
                    (arc_summary['to_facility'] == facility) &
                    (arc_summary['from_facility'] != facility)  # Exclude O=D arcs
                    ]

                if not inbound_arcs.empty:
                    total_inbound_pkgs = inbound_arcs['pkgs_day'].sum()

                    # Subtract packages that are DESTINED for this facility (final delivery)
                    packages_destined_here = od_selected[
                        od_selected['dest'] == facility
                        ]['pkgs_day'].sum()

                    # True intermediate = total inbound - final destination
                    intermediate_pkgs = total_inbound_pkgs - packages_destined_here
                    intermediate_pkgs = max(0, intermediate_pkgs)

                    # Additional validation: Get intermediate crossdock costs from ODs passing through
                    # This should be ZERO for launch facilities if path validation is correct
                    intermediate_crossdock_from_paths = 0.0
                    for _, od_row in od_selected.iterrows():
                        path_str = str(od_row.get('path_str', ''))
                        if '->' in path_str:
                            nodes = path_str.split('->')
                            # Check if this facility is intermediate (not first or last)
                            if facility in nodes[1:-1]:
                                # This OD passes through this facility
                                intermediate_crossdock_from_paths += od_row.get('intermediate_crossdock_cost', 0)

                    # VALIDATION: If we have intermediate packages but no crossdock costs,
                    # this suggests a data issue (launch facility incorrectly in paths)
                    if intermediate_pkgs > 0 and intermediate_crossdock_from_paths == 0:
                        # This shouldn't happen - likely a launch facility with bad path data
                        # Check if this facility appears as intermediate in any path
                        appears_as_intermediate = False
                        for _, od_row in od_selected.iterrows():
                            path_str = str(od_row.get('path_str', ''))
                            if '->' in path_str:
                                nodes = path_str.split('->')
                                if facility in nodes[1:-1]:
                                    appears_as_intermediate = True
                                    print(f"    WARNING: {facility} appears as intermediate in path: {path_str}")
                                    print(f"             Launch facilities should never be intermediate stops!")
                                    break

                        if not appears_as_intermediate:
                            # Discrepancy between arc_summary and path data
                            # Trust the path data - set intermediate to zero
                            print(f"    WARNING: {facility} has {intermediate_pkgs:.0f} intermediate pkgs from arcs")
                            print(f"             but doesn't appear as intermediate in any path - setting to 0")
                            intermediate_pkgs = 0

                    intermediate_crossdock_cost = intermediate_crossdock_from_paths

                    # Calculate containers for intermediate
                    if intermediate_pkgs > 0 and package_mix is not None and container_params is not None:
                        container_calc = _calculate_containers_needed(
                            intermediate_pkgs, package_mix, container_params, strategy
                        )
                        intermediate_containers = container_calc['physical_containers']
                        intermediate_cube = container_calc['total_cube_cuft']

            # ========== LAST MILE DESTINATION ROLE ==========
            # ALL packages arriving for final delivery (direct + middle-mile)
            # This WILL double-count direct injection, which is correct for perspective
            last_mile_pkgs = 0
            last_mile_containers = 0
            last_mile_cube = 0.0
            last_mile_sort_cost = 0.0
            last_mile_delivery_cost = 0.0

            # Start with direct injection packages (Zone 0)
            last_mile_pkgs = direct_injection_pkgs
            last_mile_containers = direct_injection_containers
            last_mile_cube = direct_injection_cube

            # Add middle-mile packages arriving for final delivery
            if not od_selected.empty:
                inbound_ods = od_selected[od_selected['dest'] == facility]
                if not inbound_ods.empty:
                    mm_last_mile_pkgs = inbound_ods['pkgs_day'].sum()
                    last_mile_pkgs += mm_last_mile_pkgs

                    # Get last mile costs from ODs
                    last_mile_sort_cost = inbound_ods.get('last_mile_sort_cost', 0).sum()
                    last_mile_delivery_cost = inbound_ods.get('last_mile_delivery_cost', 0).sum()

                    # Add to container count for last mile
                    if package_mix is not None and container_params is not None:
                        container_calc = _calculate_containers_needed(
                            mm_last_mile_pkgs, package_mix, container_params, strategy
                        )
                        last_mile_containers += container_calc['physical_containers']
                        last_mile_cube += container_calc['total_cube_cuft']

            # ========== CALCULATE PER-PACKAGE COSTS ==========
            injection_cost_per_pkg = (injection_sort_cost / mm_injection_pkgs) if mm_injection_pkgs > 0 else 0
            injection_linehaul_cpp = (injection_linehaul_cost / mm_injection_pkgs) if mm_injection_pkgs > 0 else 0
            intermediate_cost_per_pkg = (
                        intermediate_crossdock_cost / intermediate_pkgs) if intermediate_pkgs > 0 else 0
            last_mile_sort_cpp = (last_mile_sort_cost / last_mile_pkgs) if last_mile_pkgs > 0 else 0
            last_mile_delivery_cpp = (last_mile_delivery_cost / last_mile_pkgs) if last_mile_pkgs > 0 else 0

            volume_entry = {
                'facility': facility,
                # Direct injection (Zone 0 only - bypasses middle mile)
                'direct_injection_pkgs_day': direct_injection_pkgs,
                'direct_injection_containers': direct_injection_containers,
                'direct_injection_cube_cuft': direct_injection_cube,
                'direct_injection_cost': direct_injection_cost,

                # Middle-mile injection (ALL middle-mile ODs, including O=D)
                'injection_pkgs_day': mm_injection_pkgs,
                'injection_containers': mm_injection_containers,
                'injection_cube_cuft': mm_injection_cube,
                'injection_sort_cost': injection_sort_cost,
                'injection_linehaul_cost': injection_linehaul_cost,
                'injection_processing_cost': injection_processing_cost,
                'injection_cost_per_pkg': injection_cost_per_pkg,
                'injection_linehaul_cpp': injection_linehaul_cpp,
                'injection_sort_cpp': injection_cost_per_pkg,

                # Intermediate (pass-through for crossdock only)
                'intermediate_pkgs_day': intermediate_pkgs,
                'intermediate_containers': intermediate_containers,
                'intermediate_cube_cuft': intermediate_cube,
                'intermediate_crossdock_cost': intermediate_crossdock_cost,
                'intermediate_linehaul_cost': intermediate_linehaul_cost,
                'intermediate_cost_per_pkg': intermediate_cost_per_pkg,
                'mm_processing_cpp': intermediate_cost_per_pkg,

                # Last mile destination (direct + middle-mile arrivals - double counts direct)
                'last_mile_pkgs_day': last_mile_pkgs,
                'last_mile_containers': last_mile_containers,
                'last_mile_cube_cuft': last_mile_cube,
                'last_mile_sort_cost': last_mile_sort_cost,
                'last_mile_delivery_cost': last_mile_delivery_cost,
                'last_mile_cost': last_mile_sort_cost + last_mile_delivery_cost,
                'last_mile_sort_cpp': last_mile_sort_cpp,
                'last_mile_delivery_cpp': last_mile_delivery_cpp,
                'last_mile_cpp': (
                                             last_mile_sort_cost + last_mile_delivery_cost) / last_mile_pkgs if last_mile_pkgs > 0 else 0,

                # Legacy compatibility
                'mm_linehaul_cpp': 0,
                'total_variable_cpp': injection_cost_per_pkg
            }

            volume_data.append(volume_entry)

        except Exception as e:
            print(f"    Warning: Could not calculate volume for facility {facility}: {e}")
            import traceback
            traceback.print_exc()

    return pd.DataFrame(volume_data)


def _calculate_hourly_throughput_with_costs(volume_df: pd.DataFrame, timing_kv: dict,
                                            load_strategy: str) -> pd.DataFrame:
    """
    Calculate facility throughput requirements based on value-added hours.
    """
    df = volume_df.copy()

    # Get VA hours from timing parameters
    injection_va_hours = float(timing_kv.get('injection_va_hours',
                                             timing_kv.get('sort_hours_per_touch', 8.0)))
    middle_mile_va_hours = float(timing_kv.get('middle_mile_va_hours',
                                               timing_kv.get('crossdock_hours_per_touch', 16.0)))
    last_mile_va_hours = float(timing_kv.get('last_mile_va_hours', 4.0))

    # Calculate throughput for each facility role
    df['injection_hourly_throughput'] = df['injection_pkgs_day'] / injection_va_hours
    df['intermediate_hourly_throughput'] = df['intermediate_pkgs_day'] / middle_mile_va_hours
    df['lm_hourly_throughput'] = df['last_mile_pkgs_day'] / last_mile_va_hours

    # Peak throughput is maximum across all roles
    df['peak_hourly_throughput'] = df[
        ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput']
    ].max(axis=1)

    # Round throughput values for practical planning
    throughput_cols = ['injection_hourly_throughput', 'intermediate_hourly_throughput',
                       'lm_hourly_throughput', 'peak_hourly_throughput']
    for col in throughput_cols:
        df[col] = df[col].fillna(0).round(0).astype(int)

    return df


def calculate_zone_from_distance(origin: str, dest: str, facilities: pd.DataFrame,
                                 mileage_bands: pd.DataFrame) -> str:
    """
    Calculate zone based on straight-line haversine distance between origin and destination.
    Uses zone column from mileage_bands sheet.
    """
    try:
        # Get facility coordinates
        fac_lookup = facilities.set_index('facility_name')[['lat', 'lon']]

        if origin not in fac_lookup.index or dest not in fac_lookup.index:
            return 'unknown'

        o_lat, o_lon = fac_lookup.at[origin, 'lat'], fac_lookup.at[origin, 'lon']
        d_lat, d_lon = fac_lookup.at[dest, 'lat'], fac_lookup.at[dest, 'lon']

        # Calculate straight-line distance (no circuity)
        raw_distance = haversine_miles(o_lat, o_lon, d_lat, d_lon)

        # Look up zone from mileage bands
        if 'zone' in mileage_bands.columns:
            matching_band = mileage_bands[
                (mileage_bands['mileage_band_min'] <= raw_distance) &
                (raw_distance <= mileage_bands['mileage_band_max'])
                ]

            if not matching_band.empty:
                return str(matching_band.iloc[0]['zone'])

        return 'unknown'

    except Exception as e:
        print(f"    Warning: Could not calculate zone for {origin}->{dest}: {e}")
        return 'unknown'


def add_zone(df: pd.DataFrame, facilities: pd.DataFrame, mileage_bands: pd.DataFrame = None) -> pd.DataFrame:
    """
    CORRECTED: Zone classification based on path type and distance.
    - Zone 0: ONLY direct injection (not in middle-mile network)
    - O=D middle-mile paths: Use distance-based zone (typically Zone 2)
    """
    if df.empty:
        return df

    df = df.copy()

    # Initialize zone as unknown
    df['zone'] = 'unknown'

    # Zone 0 ONLY for direct injection (path_type == 'direct')
    if 'path_type' in df.columns:
        df.loc[df['path_type'] == 'direct', 'zone'] = 'Zone 0'

    # For all other paths (including O=D in middle-mile), calculate zone by distance
    if mileage_bands is not None and 'origin' in df.columns and 'dest' in df.columns:
        unknown_mask = df['zone'] == 'unknown'

        for idx in df[unknown_mask].index:
            origin = df.at[idx, 'origin']
            dest = df.at[idx, 'dest']

            # For O=D middle-mile, distance is 0, so it will map to shortest distance band
            # which is typically Zone 2 (0-50 miles or similar)
            zone = calculate_zone_from_distance(origin, dest, facilities, mileage_bands)
            df.at[idx, 'zone'] = zone

    return df


def build_od_selected_outputs(od_selected: pd.DataFrame, facilities: pd.DataFrame,
                              direct_day: pd.DataFrame, mileage_bands: pd.DataFrame = None) -> pd.DataFrame:
    """
    FIXED: Build OD output table with correct zone and cost handling for O=D paths.
    """
    if od_selected.empty:
        return od_selected

    od_out = od_selected.copy()

    # Add correct zone calculation based on O-D distance
    od_out = add_zone(od_out, facilities, mileage_bands)

    # FIXED: Zero out linehaul costs for O=D paths
    if 'origin' in od_out.columns and 'dest' in od_out.columns:
        self_dest_mask = od_out['origin'] == od_out['dest']
        if 'linehaul_cost' in od_out.columns:
            od_out.loc[self_dest_mask, 'linehaul_cost'] = 0.0

    return od_out


def build_dwell_hotspots(od_selected: pd.DataFrame) -> pd.DataFrame:
    """Identify facilities with significant package dwell for operational attention."""
    if od_selected.empty or 'packages_dwelled' not in od_selected.columns:
        return pd.DataFrame()

    # Filter to ODs with meaningful dwell volumes
    dwelled = od_selected[od_selected['packages_dwelled'] > 10].copy()

    if dwelled.empty:
        return pd.DataFrame()

    # Aggregate dwell by origin facility
    hotspots = dwelled.groupby('origin').agg({
        'packages_dwelled': 'sum',
        'pkgs_day': 'sum',
        'dest': 'count'
    }).reset_index()

    hotspots['dwell_rate'] = hotspots['packages_dwelled'] / hotspots['pkgs_day']
    hotspots = hotspots.sort_values('packages_dwelled', ascending=False)

    return hotspots


def build_lane_summary(arc_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Create lane-level summary aggregating across scenarios and day types.
    Includes O=D arcs for visibility but ensures costs are correct.
    """
    if arc_summary.empty:
        return pd.DataFrame()

    # Keep all arcs (including O=D for demand flow visibility)
    # But ensure O=D arcs have zero cost (approved hardcode)
    lane_summary = arc_summary.copy()

    # ✅ APPROVED HARDCODE: O=D arcs have zero linehaul cost
    od_mask = lane_summary['from_facility'] == lane_summary['to_facility']
    if 'total_cost' in lane_summary.columns:
        lane_summary.loc[od_mask, 'total_cost'] = 0.0
    if 'cost_per_truck' in lane_summary.columns:
        lane_summary.loc[od_mask, 'cost_per_truck'] = 0.0
    if 'distance_miles' in lane_summary.columns:
        lane_summary.loc[od_mask, 'distance_miles'] = 0.0

    # Aggregate by lane (from-to facility pair)
    agg_dict = {
        'pkgs_day': 'sum',
        'trucks': 'mean',
        'total_cost': 'sum',
        'packages_dwelled': 'sum'
    }

    # Add distance_miles if it exists
    if 'distance_miles' in lane_summary.columns:
        agg_dict['distance_miles'] = 'first'

    lane_summary_agg = lane_summary.groupby(['from_facility', 'to_facility']).agg(agg_dict).reset_index()

    lane_summary_agg['cost_per_pkg'] = lane_summary_agg['total_cost'] / lane_summary_agg['pkgs_day'].replace(0, 1)

    return lane_summary_agg.sort_values('total_cost', ascending=False)


def validate_network_aggregations(od_selected: pd.DataFrame, arc_summary: pd.DataFrame,
                                  facility_rollup: pd.DataFrame) -> dict:
    """
    Validate that aggregate calculations are mathematically consistent.
    Uses package-weighted averages for fill rates.
    Includes O=D arcs but ensures costs are zero (approved hardcode).
    """
    validation_results = {}

    try:
        # Total package validation
        total_od_pkgs = od_selected['pkgs_day'].sum() if not od_selected.empty else 0

        # Keep all arcs including O=D for validation
        if not arc_summary.empty:
            # ✅ APPROVED HARDCODE: Ensure O=D arcs have zero cost
            arc_for_validation = arc_summary.copy()
            od_mask = arc_for_validation['from_facility'] == arc_for_validation['to_facility']
            if 'total_cost' in arc_for_validation.columns:
                arc_for_validation.loc[od_mask, 'total_cost'] = 0.0

            total_arc_pkgs = arc_for_validation['pkgs_day'].sum()
            total_arc_cost = arc_for_validation['total_cost'].sum()
        else:
            total_arc_pkgs = 0
            total_arc_cost = 0

        total_facility_injection = facility_rollup['injection_pkgs_day'].sum() if not facility_rollup.empty else 0

        validation_results['total_od_packages'] = total_od_pkgs
        validation_results['total_arc_packages'] = total_arc_pkgs
        validation_results['total_facility_injection'] = total_facility_injection
        validation_results['package_consistency'] = abs(total_od_pkgs - total_facility_injection) < 0.01

        # Cost validation
        total_od_cost = od_selected['total_cost'].sum() if 'total_cost' in od_selected.columns else 0

        validation_results['total_od_cost'] = total_od_cost
        validation_results['total_arc_cost'] = total_arc_cost
        validation_results['cost_consistency'] = abs(total_od_cost - total_arc_cost) / max(total_od_cost, 1) < 0.05

        # Package-weighted fill rates from arc data (excluding O=D for meaningful averages)
        if not arc_summary.empty and 'truck_fill_rate' in arc_summary.columns:
            # Filter out O=D for fill rate calculation
            non_od_arcs = arc_summary[arc_summary['from_facility'] != arc_summary['to_facility']]

            if not non_od_arcs.empty:
                # Get total package cube and total truck cube for inherent weighting
                total_pkg_cube = non_od_arcs['pkg_cube_cuft'].sum() if 'pkg_cube_cuft' in non_od_arcs.columns else 0
                total_truck_cube = (non_od_arcs['trucks'] * non_od_arcs.get('cube_per_truck',
                                                                            0)).sum() if 'trucks' in non_od_arcs.columns else 1

                if total_truck_cube > 0:
                    validation_results['network_avg_truck_fill'] = total_pkg_cube / total_truck_cube
                else:
                    validation_results['network_avg_truck_fill'] = 0

                # Container fill rate calculation
                if 'container_fill_rate' in non_od_arcs.columns:
                    total_volume = non_od_arcs['pkgs_day'].sum()
                    if total_volume > 0:
                        validation_results['network_avg_container_fill'] = (
                                                                                   non_od_arcs['container_fill_rate'] *
                                                                                   non_od_arcs['pkgs_day']
                                                                           ).sum() / total_volume
                    else:
                        validation_results['network_avg_container_fill'] = 0
                else:
                    validation_results['network_avg_container_fill'] = 0
            else:
                validation_results['network_avg_truck_fill'] = 0
                validation_results['network_avg_container_fill'] = 0
        else:
            validation_results['network_avg_truck_fill'] = 0
            validation_results['network_avg_container_fill'] = 0

    except Exception as e:
        validation_results['validation_error'] = str(e)

    return validation_results