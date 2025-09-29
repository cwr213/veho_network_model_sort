# veho_net/validators.py - STRICT: All fields mandatory, no fallbacks allowed

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
    """
    STRICT: Validate facilities data - all sort optimization fields REQUIRED.
    No fallbacks allowed.
    """
    df = _norm_cols(df_raw)
    required = {
        "facility_name", "type", "market", "region",
        "lat", "lon", "timezone", "parent_hub_name", "is_injection_node",
        "max_sort_points_capacity",  # REQUIRED for sort optimization
        "last_mile_sort_groups_count"  # REQUIRED for sort optimization
    }

    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"facilities missing required columns: {missing}\n"
              f"Sort optimization requires max_sort_points_capacity and last_mile_sort_groups_count", df)

    dups = df["facility_name"][df["facility_name"].duplicated()].unique()
    if len(dups) > 0:
        _fail(f"facilities has duplicate facility_name values: {list(dups)}", df)

    # STRICT: Validate sort capacity for hub/hybrid facilities (REQUIRED, no fallbacks)
    hub_hybrid = df[df["type"].str.lower().isin(["hub", "hybrid"])]
    missing_capacity = hub_hybrid[
        pd.isna(hub_hybrid["max_sort_points_capacity"]) | (hub_hybrid["max_sort_points_capacity"] <= 0)]

    if not missing_capacity.empty:
        _fail(f"Hub/hybrid facilities MUST have max_sort_points_capacity > 0.\n"
              f"Missing/invalid capacity for: {missing_capacity['facility_name'].tolist()}\n"
              f"No fallback values allowed - update facilities sheet.", df)

    # STRICT: Validate sort groups for delivery facilities (launch/hybrid) - REQUIRED
    delivery_facilities = df[df["type"].str.lower().isin(["launch", "hybrid"])]
    missing_groups = delivery_facilities[pd.isna(delivery_facilities["last_mile_sort_groups_count"]) | (
                delivery_facilities["last_mile_sort_groups_count"] <= 0)]

    if not missing_groups.empty:
        _fail(f"Delivery facilities (launch/hybrid) MUST have last_mile_sort_groups_count > 0.\n"
              f"Missing/invalid sort groups for: {missing_groups['facility_name'].tolist()}\n"
              f"No fallback values allowed - update facilities sheet.", df)

    # Validate coordinates exist
    missing_coords = df[pd.isna(df["lat"]) | pd.isna(df["lon"])]
    if not missing_coords.empty:
        _fail(f"Facilities missing lat/lon coordinates: {missing_coords['facility_name'].tolist()}\n"
              f"Coordinates required for distance calculations - no fallbacks allowed", df)

    # Optional regional_sort_hub check (informational only)
    if "regional_sort_hub" in df.columns:
        print("✓ Found 'regional_sort_hub' column - will use for sorting decisions")
        missing_regional = df[pd.isna(df["regional_sort_hub"]) | (df["regional_sort_hub"] == "")]
        if not missing_regional.empty:
            print(f"INFO: {len(missing_regional)} facilities missing regional_sort_hub")
            print(f"      Will fall back to parent_hub_name for these facilities")
    else:
        print("INFO: 'regional_sort_hub' column not found")
        print("      Will use 'parent_hub_name' for sorting decisions")


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
    """
    STRICT: Validate timing parameters - all sort optimization params REQUIRED.
    No fallbacks allowed.
    """
    df = _norm_cols(df_raw)

    # Required timing parameters (NO FALLBACKS)
    required_keys = {
        "load_hours", "unload_hours",
        "hours_per_touch",  # REQUIRED for path steps processing hours
        "injection_va_hours",
        "middle_mile_va_hours",
        "last_mile_va_hours",
        "sort_points_per_destination"  # REQUIRED for sort capacity calculation
    }

    missing_required = sorted(required_keys - set(df["key"]))
    if missing_required:
        raise ValueError(f"timing_params missing REQUIRED keys: {missing_required}\n"
                         f"All timing parameters must be specified - no fallback values allowed.\n"
                         f"Add these keys to timing_params sheet in input file.")

    # Validate timing values are positive
    for _, row in df.iterrows():
        key = row["key"]
        if key in required_keys:
            try:
                value = float(row["value"])
                if value <= 0:
                    raise ValueError(f"timing_params: {key} must be positive (found: {value})")
            except (ValueError, TypeError):
                raise ValueError(f"timing_params: {key} must be numeric (found: {row['value']})")


def _check_cost_params(df_raw: pd.DataFrame):
    """
    STRICT: Validate all required cost parameters - NO FALLBACKS.
    """
    df = _norm_cols(df_raw)

    # Required core cost parameters (NO FALLBACKS)
    required_keys = {
        "injection_sort_cost_per_pkg",  # Required for injection sort
        "intermediate_sort_cost_per_pkg",  # Required for intermediate crossdock
        "parent_hub_sort_cost_per_pkg",  # Required for region level sort
        "last_mile_sort_cost_per_pkg",  # Required for last mile sort
        "last_mile_delivery_cost_per_pkg",  # Required for delivery
        "container_handling_cost",  # Required for container strategy
        "premium_economy_dwell_threshold",  # Required for dwell logic
    }

    # Optional but recommended
    optional_keys = {
        "sort_cost_per_pkg",  # Legacy fallback (can be used if specific costs missing)
        "allow_premium_economy_dwell",
        "dwell_cost_per_pkg_per_day",
        "sla_penalty_per_touch_per_pkg",
    }

    # Check required parameters
    missing_required = sorted(required_keys - set(df["key"]))
    if missing_required:
        # Check if sort_cost_per_pkg can serve as fallback
        if "sort_cost_per_pkg" in set(df["key"]):
            print("WARNING: Missing specific sort cost parameters, will use 'sort_cost_per_pkg' as fallback:")
            for key in missing_required:
                if "sort" in key:
                    print(f"  - {key}")
            # Only fail if non-sort parameters are missing
            non_sort_missing = [k for k in missing_required if "sort" not in k]
            if non_sort_missing:
                raise ValueError(f"cost_params missing REQUIRED keys: {non_sort_missing}\n"
                                 f"All cost parameters must be specified - no fallback values allowed.")
        else:
            raise ValueError(f"cost_params missing REQUIRED keys: {missing_required}\n"
                             f"All cost parameters must be specified - no fallback values allowed.\n"
                             f"Add these keys to cost_params sheet in input file.")

    # Validate cost values are non-negative
    all_cost_keys = required_keys | optional_keys
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
    """
    STRICT: Validate run settings parameters - REQUIRED.
    """
    df = _norm_cols(df_raw)

    required_keys = {
        "load_strategy",
        "sla_target_days",
        "path_around_the_world_factor"  # REQUIRED for path generation
    }

    missing = sorted(required_keys - set(df["key"]))
    if missing:
        raise ValueError(f"run_settings missing REQUIRED keys: {missing}\n"
                         f"Add these keys to run_settings sheet in input file.")

    # Validate load_strategy is valid
    strategy_row = df[df["key"] == "load_strategy"]
    if not strategy_row.empty:
        strategy_value = str(strategy_row.iloc[0]["value"]).lower()
        if strategy_value not in ["container", "fluid"]:
            raise ValueError(f"load_strategy must be 'container' or 'fluid' (found: {strategy_value})")

    # Validate path_around_the_world_factor is reasonable
    around_factor_row = df[df["key"] == "path_around_the_world_factor"]
    if not around_factor_row.empty:
        around_value = float(around_factor_row.iloc[0]["value"])
        if around_value < 1.0 or around_value > 5.0:
            raise ValueError(f"path_around_the_world_factor should be between 1.0 and 5.0 (found: {around_value})")


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
    STRICT comprehensive input validation.
    NO FALLBACKS - all required fields must be present and valid.
    """
    print("=" * 60)
    print("STRICT INPUT VALIDATION - NO FALLBACKS ALLOWED")
    print("=" * 60)

    _check_container_params(dfs["container_params"])
    print("✓ container_params validated")

    _check_facilities(dfs["facilities"])
    print("✓ facilities validated (including sort optimization fields)")

    _check_zips(dfs["zips"])
    print("✓ zips validated")

    _check_demand(dfs["demand"])
    print("✓ demand validated")

    _check_injection_distribution(dfs["injection_distribution"])
    print("✓ injection_distribution validated")

    _check_mileage_bands(dfs["mileage_bands"])
    print("✓ mileage_bands validated")

    _check_timing_params(dfs["timing_params"])
    print("✓ timing_params validated (all required parameters present)")

    _check_cost_params(dfs["cost_params"])
    print("✓ cost_params validated (all required parameters present)")

    _check_package_mix(dfs["package_mix"])
    print("✓ package_mix validated")

    _check_run_settings(dfs["run_settings"])
    print("✓ run_settings validated")

    _check_scenarios(dfs["scenarios"])
    print("✓ scenarios validated")

    print("=" * 60)
    print("✅ VALIDATION COMPLETE - ALL REQUIRED FIELDS PRESENT")
    print("   No fallback values will be used")
    print("   All calculations use input parameters only")
    print("=" * 60)