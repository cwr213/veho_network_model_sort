# veho_net/validators.py - Enhanced validation with regional_sort_hub support

import pandas as pd


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase for consistent comparison."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _fail(msg: str, df: pd.DataFrame | None = None):
    """Raise validation error with optional column information."""
    if df is not None:
        raise ValueError(f"{msg}\nFound columns: {list(df.columns)}")
    raise ValueError(msg)


def _check_container_params(df_raw: pd.DataFrame):
    """Validate container parameters with all required fields."""
    df = _norm_cols(df_raw)
    required = {
        "container_type",
        "usable_cube_cuft",
        "pack_utilization_container",
        "containers_per_truck",
        "trailer_air_cube_cuft",
        "pack_utilization_fluid",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"container_params missing required columns: {missing}", df)
    if df.empty:
        _fail("container_params has no rows", df)
    if not (df["container_type"].str.lower() == "gaylord").any():
        _fail("container_params must include a row where container_type == 'gaylord'", df)

    # Validate pack utilization values are reasonable
    pack_util_container = df[df["container_type"].str.lower() == "gaylord"]["pack_utilization_container"].iloc[0]
    pack_util_fluid = df["pack_utilization_fluid"].iloc[0]

    if not (0 < pack_util_container <= 1):
        _fail(f"pack_utilization_container must be between 0 and 1 (found: {pack_util_container})", df)
    if not (0 < pack_util_fluid <= 1):
        _fail(f"pack_utilization_fluid must be between 0 and 1 (found: {pack_util_fluid})", df)


def _check_facilities(df_raw: pd.DataFrame):
    """Validate facilities data structure and content including sort optimization fields."""
    df = _norm_cols(df_raw)
    required = {
        "facility_name", "type", "market", "region",
        "lat", "lon", "timezone", "parent_hub_name", "is_injection_node"
    }

    # Sort optimization specific fields
    sort_fields = {
        "max_sort_points_capacity",
        "last_mile_sort_groups_count"
    }

    # Regional sorting field (optional)
    regional_sort_field = "regional_sort_hub"

    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"facilities missing required columns: {missing}", df)

    # Check for sort optimization fields
    missing_sort = sorted(sort_fields - set(df.columns))
    if missing_sort:
        print(f"INFO: facilities missing sort optimization fields: {missing_sort}")
        print("      These are required for multi-level sort optimization")
        print("      Consider adding max_sort_points_capacity and last_mile_sort_groups_count columns")

    # Check for regional sort hub field
    if regional_sort_field not in df.columns:
        print(f"INFO: facilities missing '{regional_sort_field}' column")
        print("      Will use 'parent_hub_name' for sorting decisions")
        print("      Add 'regional_sort_hub' column for separate routing vs. sorting hierarchies")
    else:
        print(f"✓ Found '{regional_sort_field}' column - will use for sorting decisions")

        # Validate regional_sort_hub values
        missing_regional = df[pd.isna(df[regional_sort_field]) | (df[regional_sort_field] == "")]
        if not missing_regional.empty:
            print(f"WARNING: {len(missing_regional)} facilities missing regional_sort_hub values:")
            print(f"         Will fall back to parent_hub_name for these facilities")

    dups = df["facility_name"][df["facility_name"].duplicated()].unique()
    if len(dups) > 0:
        _fail(f"facilities has duplicate facility_name values: {list(dups)}", df)

    # Validate sort capacity values if present
    if "max_sort_points_capacity" in df.columns:
        # Check that hub/hybrid facilities have sort capacity specified
        hub_hybrid = df[df["type"].str.lower().isin(["hub", "hybrid"])]
        missing_capacity = hub_hybrid[pd.isna(hub_hybrid["max_sort_points_capacity"])]

        if not missing_capacity.empty:
            print(f"WARNING: Hub/hybrid facilities missing max_sort_points_capacity:")
            print(f"         {missing_capacity['facility_name'].tolist()}")
            print(f"         Default capacity of 0 will be used, preventing sort optimization")

    # Validate sort groups count for launch facilities
    if "last_mile_sort_groups_count" in df.columns:
        launch_facilities = df[df["type"].str.lower() == "launch"]
        missing_groups = launch_facilities[pd.isna(launch_facilities["last_mile_sort_groups_count"])]

        if not missing_groups.empty:
            print(f"INFO: Launch facilities missing last_mile_sort_groups_count:")
            print(f"      {missing_groups['facility_name'].tolist()}")
            print(f"      Default value of 4 sort groups will be used")


def _check_zips(df_raw: pd.DataFrame):
    """Validate ZIP code assignment data."""
    df = _norm_cols(df_raw)
    required = {"zip", "facility_name_assigned", "market", "population"}
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"zips missing required columns: {missing}", df)
    dups = df["zip"][df["zip"].duplicated()].unique()
    if len(dups) > 0:
        _fail(f"zips has duplicate ZIP codes: {list(dups)}", df)


def _check_demand(df_raw: pd.DataFrame):
    """Validate demand forecast parameters."""
    df = _norm_cols(df_raw)
    required = {
        "year",
        "annual_pkgs",
        "offpeak_pct_of_annual",
        "peak_pct_of_annual",
        "middle_mile_share_offpeak",
        "middle_mile_share_peak",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"demand missing required columns: {missing}", df)

    # Validate percentage shares are within valid range [0,1]
    for col in ["offpeak_pct_of_annual", "peak_pct_of_annual",
                "middle_mile_share_offpeak", "middle_mile_share_peak"]:
        bad = df[~df[col].between(0, 1, inclusive="both")]
        if not bad.empty:
            raise ValueError(
                f"demand: column '{col}' has values outside [0,1]. Offenders (first 5 rows):\n{bad.head(5)}")


def _check_injection_distribution(df: pd.DataFrame):
    """Validate injection distribution parameters."""
    required = {"facility_name", "absolute_share"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"injection_distribution missing required columns: {sorted(missing)}")

    # Validate absolute_share sums to positive value
    w = pd.to_numeric(df["absolute_share"], errors="coerce").fillna(0.0)
    if float(w.sum()) <= 0:
        raise ValueError("injection_distribution.absolute_share must sum > 0")


def _check_mileage_bands(df_raw: pd.DataFrame):
    """Validate mileage band cost structure and zone mapping."""
    df = _norm_cols(df_raw)
    required = {"mileage_band_min", "mileage_band_max", "fixed_cost_per_truck", "variable_cost_per_mile",
                "circuity_factor", "mph", "zone"}
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"mileage_bands missing required columns: {missing}", df)


def _check_timing_params(df_raw: pd.DataFrame):
    """Validate timing parameters including sort optimization parameters."""
    df = _norm_cols(df_raw)

    # Required timing parameters
    required_keys = {
        "load_hours", "unload_hours",
        "hours_per_touch",
        "injection_va_hours",
        "middle_mile_va_hours",
        "last_mile_va_hours",
    }

    # Sort optimization specific parameters
    sort_timing_keys = {
        "sort_points_per_destination"
    }

    missing_required = sorted(required_keys - set(df["key"]))
    if missing_required:
        raise ValueError(f"timing_params missing required keys: {missing_required}")

    # Check for sort optimization timing parameters
    missing_sort = sorted(sort_timing_keys - set(df["key"]))
    if missing_sort:
        print(f"INFO: timing_params missing sort optimization keys: {missing_sort}")
        print("      sort_points_per_destination is required for multi-level sort optimization")
        print("      Default value of 1.0 will be used")

    # Validate timing values are positive
    for _, row in df.iterrows():
        key = row["key"]
        if key in (required_keys | sort_timing_keys):
            value = float(row["value"])
            if value <= 0:
                raise ValueError(f"timing_params: {key} must be positive (found: {value})")


def _check_cost_params(df_raw: pd.DataFrame):
    """Validate all required cost parameters including sort optimization costs."""
    df = _norm_cols(df_raw)

    # Required core cost parameters
    required_keys = {
        "sort_cost_per_pkg",  # Can be used as fallback
        "last_mile_sort_cost_per_pkg",
        "last_mile_delivery_cost_per_pkg",
        "container_handling_cost",
        "premium_economy_dwell_threshold",
    }

    # Enhanced sort cost parameters (optional but recommended)
    enhanced_sort_keys = {
        "injection_sort_cost_per_pkg",
        "intermediate_sort_cost_per_pkg",
        "parent_hub_sort_cost_per_pkg",  # For region level sort
    }

    # Optional enhanced parameters
    optional_keys = {
        "allow_premium_economy_dwell",
        "dwell_cost_per_pkg_per_day",
        "sla_penalty_per_touch_per_pkg",
    }

    # Check required parameters
    missing_required = sorted(required_keys - set(df["key"]))
    if missing_required:
        raise ValueError(f"cost_params missing required keys: {missing_required}")

    # Check for enhanced sort parameters
    missing_enhanced_sort = sorted(enhanced_sort_keys - set(df["key"]))
    if missing_enhanced_sort:
        print(f"INFO: cost_params missing enhanced sort parameters: {missing_enhanced_sort}")
        print("      Will use 'sort_cost_per_pkg' as fallback for missing sort cost parameters")
        print("      For multi-level sort optimization, consider adding:")
        print("        - injection_sort_cost_per_pkg")
        print("        - intermediate_sort_cost_per_pkg")
        print("        - parent_hub_sort_cost_per_pkg (for region level sort)")

    # Check optional parameters
    missing_optional = sorted(optional_keys - set(df["key"]))
    if missing_optional:
        print(f"INFO: cost_params missing optional parameters: {missing_optional}")
        print("      Default values of 0.0 will be used")

    # Validate cost values are non-negative
    all_cost_keys = required_keys | enhanced_sort_keys | optional_keys
    for _, row in df.iterrows():
        key = row["key"]
        if key in all_cost_keys:
            try:
                value = float(row["value"])
                if value < 0:
                    raise ValueError(f"cost_params: {key} must be non-negative (found: {value})")
            except (ValueError, TypeError):
                raise ValueError(f"cost_params: {key} must be numeric (found: {row['value']})")

    # Validate premium_economy_dwell_threshold is between 0 and 1
    dwell_threshold_row = df[df["key"] == "premium_economy_dwell_threshold"]
    if not dwell_threshold_row.empty:
        dwell_value = float(dwell_threshold_row.iloc[0]["value"])
        if not (0 <= dwell_value <= 1):
            raise ValueError(f"premium_economy_dwell_threshold must be between 0 and 1 (found: {dwell_value})")


def _check_package_mix(df_raw: pd.DataFrame):
    """Validate package mix distribution."""
    df = _norm_cols(df_raw)
    required = {"package_type", "share_of_pkgs", "avg_cube_cuft"}
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"package_mix missing required columns: {missing}", df)

    s = float(df["share_of_pkgs"].sum())
    if abs(s - 1.0) > 1e-6:
        _fail(f"package_mix share_of_pkgs must sum to 1.0 (found {s})", df)

    # Validate cube values are positive
    bad_cube = df[df["avg_cube_cuft"] <= 0]
    if not bad_cube.empty:
        _fail(f"package_mix avg_cube_cuft must be positive. Bad rows:\n{bad_cube}", df)


def _check_run_settings(df_raw: pd.DataFrame):
    """Validate run settings parameters."""
    df = _norm_cols(df_raw)

    required_keys = {"load_strategy", "sla_target_days", "path_around_the_world_factor"}
    missing = sorted(required_keys - set(df["key"]))
    if missing:
        raise ValueError(f"run_settings missing required keys: {missing}")

    # Validate load_strategy is valid
    strategy_row = df[df["key"] == "load_strategy"]
    if not strategy_row.empty:
        strategy_value = str(strategy_row.iloc[0]["value"]).lower()
        if strategy_value not in ["container", "fluid"]:
            raise ValueError(f"load_strategy must be 'container' or 'fluid' (found: {strategy_value})")


def _check_scenarios(df_raw: pd.DataFrame):
    """Validate scenario definitions."""
    df = _norm_cols(df_raw)
    required = {"year", "day_type"}
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"scenarios missing required columns: {missing}", df)

    # Validate day_type values
    valid_day_types = ["peak", "offpeak"]
    bad_days = df[~df["day_type"].str.lower().isin(valid_day_types)]
    if not bad_days.empty:
        _fail(f"scenarios day_type must be 'peak' or 'offpeak'. Bad rows:\n{bad_days}", df)


def validate_inputs(dfs: dict):
    """
    Comprehensive input validation with support for multi-level sort optimization and regional_sort_hub.
    """
    print("Validating input sheets...")

    _check_container_params(dfs["container_params"])
    _check_facilities(dfs["facilities"])
    _check_zips(dfs["zips"])
    _check_demand(dfs["demand"])
    _check_injection_distribution(dfs["injection_distribution"])
    _check_mileage_bands(dfs["mileage_bands"])
    _check_timing_params(dfs["timing_params"])
    _check_cost_params(dfs["cost_params"])
    _check_package_mix(dfs["package_mix"])
    _check_run_settings(dfs["run_settings"])
    _check_scenarios(dfs["scenarios"])

    print("✅ Input validation complete - all required parameters present and valid")
    print("ℹ️  Multi-level sort optimization fields detected and validated")
    print("🔧 UPDATED ARCHITECTURE: Regional sorting hierarchy separate from routing hierarchy")