"""
Automated experiment runner for the parallel PrivMAS workflow.

This script loops through a range of agent counts, runs the parallel evaluation
for each count, and prints a summary of the results.
"""
import sys
import os
import argparse
import pandas as pd
import json
import time

# Add the project root to the Python path to allow importing from main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import run_parallel_privmas_evaluation
from visualization.reporting import generate_report

def run_experiments(max_agents: int, max_rows: int):
    """
    Runs the parallel workflow with a varying number of agents and collects results.
    """
    all_run_details = []
    
    # Load test data once
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'privmas_dataset_1000.json')
    try:
        with open(data_path, 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test data not found at {data_path}")
        return

    print(f"Running experiments for 1 to {max_agents} agents...")
    
    for i in range(1, max_agents + 1):
        print(f"\n--- Running with {i} agent(s) ---")
        
        run_df = run_parallel_privmas_evaluation(
            config_path="config.yaml",
            max_rows=max_rows,
            num_agents=i,
            quiet=True,
            test_data=test_data,
        )
        
        # Add num_agents to each row for detailed analysis
        run_df['num_agents'] = i
        all_run_details.append(run_df)

        total_time_ms = run_df['e2e_ms'].sum()
        avg_e2e_ms = run_df['e2e_ms'].mean() if not run_df.empty else 0
        print(f"Completed run for {i} agent(s) in {total_time_ms:.2f} ms. Avg E2E/row: {avg_e2e_ms:.2f} ms.")

    if not all_run_details:
        print("No results to summarize.")
        return

    # Combine all results into a single DataFrame
    full_results_df = pd.concat(all_run_details, ignore_index=True)

    # --- Correctly calculate overall metrics ---
    def calculate_metrics_from_counts(df, prefix):
        total_tp = df[f'{prefix}_tp'].sum()
        total_fp = df[f'{prefix}_fp'].sum()
        total_fn = df[f'{prefix}_fn'].sum()
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return pd.Series({
            f'{prefix}_precision': precision,
            f'{prefix}_recall': recall,
            f'{prefix}_f1': f1,
        })

    # Group by number of agents and apply the correct calculation for both strict and overlap
    strict_summary = full_results_df.groupby('num_agents').apply(lambda df: calculate_metrics_from_counts(df, 'strict')).reset_index()
    overlap_summary = full_results_df.groupby('num_agents').apply(lambda df: calculate_metrics_from_counts(df, 'overlap')).reset_index()

    # Also get the timing and leakage summaries
    other_summary = full_results_df.groupby('num_agents').agg(
        avg_e2e_ms=('e2e_ms', 'mean'),
        avg_leakage_rate=('leakage_rate', 'mean'),
        total_runtime_ms=('e2e_ms', 'sum')
    ).reset_index()

    # Merge the summaries
    summary_table = pd.merge(other_summary, strict_summary, on='num_agents')
    summary_table = pd.merge(summary_table, overlap_summary, on='num_agents')


    print("\n=== Experiment Summary ===")
    # Define columns to display
    display_cols = [
        'num_agents', 'avg_e2e_ms', 
        'strict_f1', 'overlap_f1', 'avg_leakage_rate',
        'strict_precision', 'strict_recall',
        'overlap_precision', 'overlap_recall',
        'total_runtime_ms'
    ]
    print(summary_table[display_cols].round(3))

    # Generate detailed report with plots
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiment_report')
    
    # We need to structure the data for the report generator
    report_data = []
    for num_agents, group in full_results_df.groupby('num_agents'):
        # Calculate micro-averaged metrics for the report, focusing on overlap
        report_precision = group['overlap_tp'].sum() / (group['overlap_tp'].sum() + group['overlap_fp'].sum()) if (group['overlap_tp'].sum() + group['overlap_fp'].sum()) > 0 else 0
        report_recall = group['overlap_tp'].sum() / (group['overlap_tp'].sum() + group['overlap_fn'].sum()) if (group['overlap_tp'].sum() + group['overlap_fn'].sum()) > 0 else 0
        report_f1 = 2 * (report_precision * report_recall) / (report_precision + report_recall) if (report_precision + report_recall) > 0 else 0

        report_data.append({
            'num_agents': num_agents,
            'e2e_ms': group['e2e_ms'].mean(),
            't_inf_ms': group['t_inf_ms'].mean(),
            'delta_sync_ms': group['delta_sync_ms'].mean(),
            'c_tax_ms': group['c_tax_ms'].mean(),
            'agent_details': group['agent_details'].tolist(),
            'precision': report_precision,
            'recall': report_recall,
            'f1_score': report_f1,
            'leakage_rate': group['leakage_rate'].mean(),
            # Note: accuracy_by_label is no longer generated in the same way
            'accuracy_by_label': [], 
        })

    generate_report(report_data, output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel PrivMAS experiments.")
    parser.add_argument("--max-agents", type=int, default=8, help="Maximum number of agents to test.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for the dataset.")
    
    args = parser.parse_args()
    
    run_experiments(args.max_agents, args.max_rows)

