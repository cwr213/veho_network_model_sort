import pandas as pd
import numpy as np
from pathlib import Path


def safe_sheet_name(name: str) -> str:
    """Create Excel-safe sheet names that are <= 31 characters."""
    invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
    clean_name = name
    for char in invalid_chars:
        clean_name = clean_name.replace(char, '_')
    return clean_name[:31]


def write_workbook(path, scenario_summary, od_out, path_detail, dwell_hotspots, facility_rollup, arc_summary, kpis,
                   sort_allocation_summary=None):
    """Write simplified workbook with core sheets only (backward compatibility)."""
    return write_workbook_with_sort_analysis(
        path, scenario_summary, od_out, path_detail, dwell_hotspots,
        facility_rollup, arc_summary, kpis, pd.DataFrame()
    )


def write_workbook_with_sort_analysis(path, scenario_summary, od_out, path_detail, dwell_hotspots, facility_rollup,
                                      arc_summary, kpis, sort_summary=None):
    """EXTENDED: Write workbook with multi-level sort analysis included."""
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:

            # Core sheets matching expected output structure

            # 1. Scenario Summary (key-value format)
            if not scenario_summary.empty:
                scenario_summary.to_excel(xw, sheet_name="scenario_summary", index=False)
            else:
                pd.DataFrame([{"key": "scenario_id", "value": "unknown"}]).to_excel(
                    xw, sheet_name="scenario_summary", index=False)

            # 2. OD Selected Paths
            if not od_out.empty:
                od_out.to_excel(xw, sheet_name="od_selected_paths", index=False)
            else:
                pd.DataFrame([{"note": "No OD path data"}]).to_excel(
                    xw, sheet_name="od_selected_paths", index=False)

            # 3. Path Steps Selected
            if not path_detail.empty:
                path_detail.to_excel(xw, sheet_name="path_steps_selected", index=False)
            else:
                pd.DataFrame([{"note": "No path detail data"}]).to_excel(
                    xw, sheet_name="path_steps_selected", index=False)

            # 4. Dwell Hotspots
            if not dwell_hotspots.empty:
                dwell_hotspots.to_excel(xw, sheet_name="dwell_hotspots", index=False)
            else:
                pd.DataFrame([{"note": "No dwell hotspot data"}]).to_excel(
                    xw, sheet_name="dwell_hotspots", index=False)

            # 5. Facility Rollup
            if not facility_rollup.empty:
                facility_rollup.to_excel(xw, sheet_name="facility_rollup", index=False)
            else:
                pd.DataFrame([{"note": "No facility data"}]).to_excel(
                    xw, sheet_name="facility_rollup", index=False)

            # 6. Arc Summary
            if not arc_summary.empty:
                arc_summary.to_excel(xw, sheet_name="arc_summary", index=False)
            else:
                pd.DataFrame([{"note": "No arc/lane data"}]).to_excel(
                    xw, sheet_name="arc_summary", index=False)

            # 7. KPIs (key-value format)
            if kpis is not None and not kpis.empty:
                kpis.to_frame("value").to_excel(xw, sheet_name="kpis")
            else:
                pd.Series([0], index=["total_cost"]).to_frame("value").to_excel(
                    xw, sheet_name="kpis")

            # 8. EXTENDED: Sort Level Analysis
            if sort_summary is not None and not sort_summary.empty:
                # Main sort summary
                sort_summary.to_excel(xw, sheet_name="sort_analysis", index=False)

                # Sort level summary statistics with baseline comparison
                baseline_metrics = None
                if kpis is not None:
                    baseline_metrics = {
                        'baseline_total_cost': kpis.get('baseline_total_cost', 0),
                        'constraint_cost_impact': kpis.get('constraint_cost_impact', 0),
                        'constraint_cost_impact_pct': kpis.get('constraint_cost_impact_pct', 0)
                    }

                sort_stats = create_sort_level_summary_stats(sort_summary, baseline_metrics)
                sort_stats.to_excel(xw, sheet_name="sort_level_stats", index=False)

                # EXTENDED: Baseline vs Optimized Comparison
                if baseline_metrics and any(baseline_metrics.values()):
                    baseline_comparison = pd.DataFrame([
                        {
                            'metric': 'Total Daily Cost',
                            'baseline_unconstrained': baseline_metrics['baseline_total_cost'],
                            'optimized_constrained': kpis.get('optimized_total_cost', 0),
                            'difference': baseline_metrics['constraint_cost_impact'],
                            'difference_pct': baseline_metrics['constraint_cost_impact_pct'],
                            'explanation': 'Impact of sort capacity constraints'
                        },
                        {
                            'metric': 'Cost per Package',
                            'baseline_unconstrained': kpis.get('baseline_cost_per_pkg', 0),
                            'optimized_constrained': kpis.get('optimized_cost_per_pkg', 0),
                            'difference': kpis.get('optimized_cost_per_pkg', 0) - kpis.get('baseline_cost_per_pkg', 0),
                            'difference_pct': ((kpis.get('optimized_cost_per_pkg', 0) - kpis.get(
                                'baseline_cost_per_pkg', 0)) / max(kpis.get('baseline_cost_per_pkg', 1), 1)) * 100,
                            'explanation': 'Per-package impact of capacity constraints'
                        }
                    ])
                    baseline_comparison.to_excel(xw, sheet_name="baseline_comparison", index=False)

                # Sort capacity utilization
                sort_capacity = create_sort_capacity_analysis(sort_summary, facility_rollup)
                if not sort_capacity.empty:
                    sort_capacity.to_excel(xw, sheet_name="sort_capacity", index=False)
            else:
                pd.DataFrame([{"note": "No sort analysis data"}]).to_excel(
                    xw, sheet_name="sort_analysis", index=False)

        return True

    except Exception as e:
        print(f"Error writing workbook {path}: {e}")
        return False


def create_sort_level_summary_stats(sort_summary: pd.DataFrame, baseline_metrics: dict = None) -> pd.DataFrame:
    """Create summary statistics for sort level decisions with baseline comparison."""
    try:
        if sort_summary.empty:
            return pd.DataFrame()

        # Overall statistics
        total_ods = len(sort_summary)
        total_volume = sort_summary['pkgs_day'].sum()
        total_cost = sort_summary['total_sort_cost'].sum()
        total_savings = sort_summary['savings_vs_market'].sum()

        # By sort level
        sort_level_stats = sort_summary.groupby('chosen_sort_level').agg({
            'pkgs_day': ['count', 'sum'],
            'total_sort_cost': 'sum',
            'savings_vs_market': 'sum'
        }).round(2)

        sort_level_stats.columns = ['od_count', 'total_volume', 'total_cost', 'total_savings']
        sort_level_stats = sort_level_stats.reset_index()

        # Add percentages
        sort_level_stats['pct_of_ods'] = (sort_level_stats['od_count'] / total_ods * 100).round(1)
        sort_level_stats['pct_of_volume'] = (sort_level_stats['total_volume'] / total_volume * 100).round(1)
        sort_level_stats['avg_cost_per_pkg'] = (
                    sort_level_stats['total_cost'] / sort_level_stats['total_volume']).round(3)

        # Summary row with baseline comparison if provided
        summary_data = {
            'chosen_sort_level': 'TOTAL',
            'od_count': total_ods,
            'total_volume': total_volume,
            'total_cost': total_cost,
            'total_savings': total_savings,
            'pct_of_ods': 100.0,
            'pct_of_volume': 100.0,
            'avg_cost_per_pkg': total_cost / total_volume if total_volume > 0 else 0
        }

        # Add baseline comparison if available
        if baseline_metrics:
            summary_data.update({
                'baseline_total_cost': baseline_metrics.get('baseline_total_cost', 0),
                'constraint_cost_impact': baseline_metrics.get('constraint_cost_impact', 0),
                'constraint_impact_pct': baseline_metrics.get('constraint_cost_impact_pct', 0)
            })

        summary_row = pd.DataFrame([summary_data])

        return pd.concat([sort_level_stats, summary_row], ignore_index=True)

    except Exception as e:
        return pd.DataFrame([{"error": f"Could not create sort stats: {e}"}])


def create_sort_capacity_analysis(sort_summary: pd.DataFrame, facility_rollup: pd.DataFrame) -> pd.DataFrame:
    """Create analysis of sort capacity utilization by facility."""
    try:
        if sort_summary.empty or facility_rollup.empty:
            return pd.DataFrame()

        # Aggregate sort requirements by origin facility
        sort_requirements = sort_summary.groupby('origin').agg({
            'pkgs_day': 'sum',
            'chosen_sort_level': lambda x: x.value_counts().to_dict()
        }).reset_index()

        # Join with facility data
        capacity_analysis = sort_requirements.merge(
            facility_rollup[['facility', 'injection_pkgs_day', 'peak_hourly_throughput']],
            left_on='origin',
            right_on='facility',
            how='left'
        )

        # Calculate estimated sort point usage (simplified)
        capacity_analysis['estimated_sort_points'] = capacity_analysis['pkgs_day'] / 1000  # Simplified estimate

        return capacity_analysis.drop(columns=['facility'])

    except Exception as e:
        return pd.DataFrame([{"error": f"Could not create capacity analysis: {e}"}])


def write_compare_workbook(path, compare_df: pd.DataFrame, run_kv: dict):
    """Write comparison workbook with expected sheets."""
    try:
        # Ensure required columns exist
        required_cols = ['scenario_id', 'strategy', 'total_cost', 'cost_per_pkg']
        missing_cols = [col for col in required_cols if col not in compare_df.columns]
        if missing_cols:
            print(f"Missing columns in compare_df: {missing_cols}")
            return False

        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:

            # Long format comparison
            compare_df.to_excel(xw, sheet_name="kpi_compare_long", index=False)

            # Wide format comparison (pivot by strategy)
            try:
                metrics_to_pivot = ['total_cost', 'cost_per_pkg', 'total_sort_savings']
                available_metrics = [col for col in metrics_to_pivot if col in compare_df.columns]

                if available_metrics and 'strategy' in compare_df.columns:
                    # Create identifier for pivoting
                    if 'scenario_id_from_input' in compare_df.columns:
                        pivot_id = compare_df['scenario_id_from_input']
                    elif 'base_id' in compare_df.columns:
                        pivot_id = compare_df['base_id']
                    else:
                        pivot_id = compare_df['scenario_id']

                    wide_df = compare_df.pivot_table(
                        index=pivot_id.name if hasattr(pivot_id, 'name') else 'id',
                        columns='strategy',
                        values=available_metrics,
                        aggfunc='first'
                    )
                    wide_df.to_excel(xw, sheet_name="kpi_compare_wide")
                else:
                    # Fallback: duplicate long format
                    compare_df.to_excel(xw, sheet_name="kpi_compare_wide", index=False)

            except Exception as e:
                print(f"Warning: Could not create wide format comparison: {e}")
                # Use long format as fallback
                compare_df.to_excel(xw, sheet_name="kpi_compare_wide", index=False)

            # EXTENDED: Sort level comparison if available
            sort_cols = ['pct_region_sort', 'pct_market_sort', 'pct_sort_group_sort', 'total_sort_savings']
            available_sort_cols = [col for col in sort_cols if col in compare_df.columns]

            if available_sort_cols:
                sort_comparison = compare_df[['scenario_id', 'strategy'] + available_sort_cols]
                sort_comparison.to_excel(xw, sheet_name="sort_comparison", index=False)

            # Run settings
            settings = pd.DataFrame([{"key": k, "value": v} for k, v in run_kv.items()])
            settings.to_excel(xw, sheet_name="run_settings", index=False)

        return True

    except Exception as e:
        print(f"Error writing comparison workbook {path}: {e}")
        return False


def write_executive_summary_workbook(path, results_by_strategy: dict, run_kv: dict, base_id: str):
    """Write executive summary with key business insights including sort optimization."""
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:

            # Strategy Comparison (updated with sort metrics)
            strategy_comparison = create_enhanced_strategy_comparison(results_by_strategy)
            strategy_comparison.to_excel(xw, sheet_name="Strategy_Comparison", index=False)

            # Hub Hourly Throughput
            hub_throughput = create_hub_throughput_analysis(results_by_strategy)
            if not hub_throughput.empty:
                hub_throughput.to_excel(xw, sheet_name="Hub_Hourly_Throughput", index=False)

            # EXTENDED: Sort Level Optimization Summary
            sort_optimization_summary = create_sort_optimization_executive_summary(results_by_strategy)
            if not sort_optimization_summary.empty:
                sort_optimization_summary.to_excel(xw, sheet_name="Sort_Optimization", index=False)

            # Facility Truck Requirements
            truck_requirements = create_facility_truck_requirements(results_by_strategy)
            if not truck_requirements.empty:
                truck_requirements.to_excel(xw, sheet_name="Facility_Truck_Requirements", index=False)

            # Path Type Analysis
            path_analysis = create_path_type_analysis(results_by_strategy)
            if not path_analysis.empty:
                path_analysis.to_excel(xw, sheet_name="Path_Type_Analysis", index=False)

            # EXTENDED: Key Answers (updated with sort insights)
            key_answers = create_enhanced_key_answers(results_by_strategy)
            key_answers.to_excel(xw, sheet_name="Key_Answers", index=False)

        return True

    except Exception as e:
        print(f"Error writing executive summary {path}: {e}")
        return False


def create_enhanced_strategy_comparison(results_by_strategy: dict) -> pd.DataFrame:
    """Create strategy comparison table including sort optimization metrics."""
    try:
        comparison_data = []

        for strategy, data in results_by_strategy.items():
            kpis = data.get('kpis', pd.Series())

            comparison_data.append({
                'strategy': strategy,
                'total_cost': kpis.get('total_cost', 0),
                'cost_per_pkg': kpis.get('cost_per_pkg', 0),
                'num_ods': kpis.get('num_ods', 0),
                'avg_truck_fill_rate': kpis.get('avg_truck_fill_rate', 0),
                'avg_container_fill_rate': kpis.get('avg_container_fill_rate', 0),
                'pct_direct': kpis.get('pct_direct', 0),
                'pct_1_touch': kpis.get('pct_1_touch', 0),
                'pct_2_touch': kpis.get('pct_2_touch', 0),
                # EXTENDED: Sort optimization metrics
                'pct_region_sort': kpis.get('pct_region_sort', 0),
                'pct_market_sort': kpis.get('pct_market_sort', 0),
                'pct_sort_group_sort': kpis.get('pct_sort_group_sort', 0),
                'total_sort_savings': kpis.get('total_sort_savings', 0)
            })

        return pd.DataFrame(comparison_data)

    except Exception as e:
        return pd.DataFrame([{"error": f"Could not create strategy comparison: {e}"}])


def create_sort_optimization_executive_summary(results_by_strategy: dict) -> pd.DataFrame:
    """Create executive summary of sort optimization results."""
    try:
        summary_data = []

        for strategy, data in results_by_strategy.items():
            kpis = data.get('kpis', pd.Series())

            # Calculate sort level distribution
            total_ods = kpis.get('num_ods', 0)
            region_pct = kpis.get('pct_region_sort', 0)
            market_pct = kpis.get('pct_market_sort', 0)
            sort_group_pct = kpis.get('pct_sort_group_sort', 0)
            total_savings = kpis.get('total_sort_savings', 0)

            summary_data.append({
                'strategy': strategy,
                'total_od_pairs': total_ods,
                'region_sort_ods': int(total_ods * region_pct / 100),
                'market_sort_ods': int(total_ods * market_pct / 100),
                'sort_group_ods': int(total_ods * sort_group_pct / 100),
                'region_sort_pct': region_pct,
                'market_sort_pct': market_pct,
                'sort_group_pct': sort_group_pct,
                'total_daily_savings': total_savings,
                'annual_savings_estimate': total_savings * 365,
                'recommendation': determine_sort_recommendation(region_pct, market_pct, sort_group_pct, total_savings)
            })

        return pd.DataFrame(summary_data)

    except Exception as e:
        return pd.DataFrame([{"error": f"Could not create sort summary: {e}"}])


def determine_sort_recommendation(region_pct, market_pct, sort_group_pct, constraint_impact):
    """Generate sort level recommendation based on optimization results and constraint impact."""
    if abs(constraint_impact) > 50000:  # Significant constraint impact
        if region_pct > 50:
            return "Capacity constraints forcing region-level sorting - consider capacity expansion"
        elif sort_group_pct < 5:
            return "Sort group level unavailable due to capacity limits - expand sort capacity"
        else:
            return "Mixed strategy driven by capacity constraints"
    else:
        if sort_group_pct > 50:
            return "Sort group level optimization successful - pursue granular sorting"
        elif region_pct > 30:
            return "Region-level sorting economical - implement consolidation strategy"
        else:
            return "Market-level sorting remains optimal under current constraints"


def create_hub_throughput_analysis(results_by_strategy: dict) -> pd.DataFrame:
    """Create hub throughput analysis."""
    try:
        # Get container strategy facility data
        container_data = results_by_strategy.get('container', {})
        facility_rollup = container_data.get('facility_rollup', pd.DataFrame())

        if facility_rollup.empty:
            return pd.DataFrame()

        # Filter to hubs and include throughput metrics
        hub_data = facility_rollup[
            facility_rollup.get('type', '').isin(['hub', 'hybrid'])
        ].copy()

        if hub_data.empty:
            return pd.DataFrame()

        throughput_cols = ['facility', 'peak_hourly_throughput', 'injection_pkgs_day', 'last_mile_pkgs_day']
        available_cols = [col for col in throughput_cols if col in hub_data.columns]

        return hub_data[available_cols]

    except Exception:
        return pd.DataFrame()


def create_facility_truck_requirements(results_by_strategy: dict) -> pd.DataFrame:
    """Create facility truck requirements analysis."""
    try:
        truck_data = []

        for strategy, data in results_by_strategy.items():
            facility_rollup = data.get('facility_rollup', pd.DataFrame())

            if not facility_rollup.empty:
                for _, facility in facility_rollup.iterrows():
                    truck_data.append({
                        'facility': facility.get('facility', 'unknown'),
                        'strategy': strategy,
                        'peak_throughput': facility.get('peak_hourly_throughput', 0),
                        'estimated_trucks_needed': max(1, int(facility.get('peak_hourly_throughput', 0) / 100))
                    })

        return pd.DataFrame(truck_data)

    except Exception:
        return pd.DataFrame()


def create_path_type_analysis(results_by_strategy: dict) -> pd.DataFrame:
    """Create path type analysis."""
    try:
        path_analysis = []

        for strategy, data in results_by_strategy.items():
            kpis = data.get('kpis', pd.Series())

            path_analysis.append({
                'strategy': strategy,
                'pct_direct': kpis.get('pct_direct', 0),
                'pct_1_touch': kpis.get('pct_1_touch', 0),
                'pct_2_touch': kpis.get('pct_2_touch', 0),
                'pct_3_touch': kpis.get('pct_3_touch', 0)
            })

        return pd.DataFrame(path_analysis)

    except Exception:
        return pd.DataFrame()


def create_enhanced_key_answers(results_by_strategy: dict) -> pd.DataFrame:
    """Create key answers including sort optimization insights and baseline comparison."""
    try:
        container_kpis = results_by_strategy.get('container', {}).get('kpis', pd.Series())

        container_cost = container_kpis.get('total_cost', 0)
        baseline_cost = container_kpis.get('baseline_total_cost', 0)
        constraint_impact = container_kpis.get('constraint_cost_impact', 0)
        constraint_impact_pct = container_kpis.get('constraint_cost_impact_pct', 0)

        # Sort level distribution
        region_pct = container_kpis.get('pct_region_sort', 0)
        market_pct = container_kpis.get('pct_market_sort', 0)
        sort_group_pct = container_kpis.get('pct_sort_group_sort', 0)

        key_answers = pd.DataFrame([
            {
                'question': '1. Optimal Network Cost (Constrained)',
                'answer': f'Total daily cost: ${container_cost:,.0f}',
                'detail': f'Cost per package: ${container_kpis.get("cost_per_pkg", 0):.3f}',
                'metric': f'Serving {container_kpis.get("num_ods", 0)} OD pairs'
            },
            {
                'question': '2. Baseline vs. Constrained Comparison',
                'answer': f'Unconstrained baseline: ${baseline_cost:,.0f}',
                'detail': f'Capacity constraint impact: ${constraint_impact:,.0f} ({constraint_impact_pct:+.1f}%)',
                'metric': f'Constraint cost per package: ${constraint_impact / max(container_kpis.get("num_ods", 1), 1):.3f}'
            },
            {
                'question': '3. Sort Level Optimization Results',
                'answer': determine_sort_recommendation(region_pct, market_pct, sort_group_pct, constraint_impact),
                'detail': f'Distribution: Region {region_pct:.1f}%, Market {market_pct:.1f}%, Sort Group {sort_group_pct:.1f}%',
                'metric': f'Capacity constraints drove {region_pct:.1f}% to region-level sorting'
            },
            {
                'question': '4. Key Business Insights',
                'answer': f'Capacity constraints increase costs by {constraint_impact_pct:.1f}%',
                'detail': f'Fill rates: Truck {container_kpis.get("avg_truck_fill_rate", 0):.1%}, Container {container_kpis.get("avg_container_fill_rate", 0):.1%}',
                'metric': 'Consider capacity expansion if cost impact exceeds acceptable threshold'
            }
        ])

        return key_answers

    except Exception as e:
        return pd.DataFrame([{"question": "Error", "answer": "Could not generate answers", "detail": str(e)}])


def write_consolidated_multi_year_workbook(path: Path, all_results: list, run_kv: dict):
    """Write consolidated workbook - simplified version."""
    try:
        consolidated_data = {}

        # Just collect key data for now
        for result in all_results:
            scenario_id = result.get('scenario_id', 'unknown')
            consolidated_data[scenario_id] = {
                'total_cost': result.get('total_cost', 0),
                'strategy': result.get('strategy', 'unknown'),
                'total_sort_savings': result.get('total_sort_savings', 0)
            }

        summary_df = pd.DataFrame([
            {'scenario_id': k, **v} for k, v in consolidated_data.items()
        ])

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, sheet_name="Consolidated_Summary", index=False)

        return True

    except Exception:
        return False