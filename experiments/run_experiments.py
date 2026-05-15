"""
Automated experiment runner for the parallel PrivMAS workflow.

This script loops through a range of agent counts, runs the parallel evaluation
for each count, and prints a summary of the results.
"""
import sys
import os
import argparse
import pandas as pd
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
    
    print(f"Running experiments for 1 to {max_agents} agents...")
    
    for i in range(1, max_agents + 1):
        print(f"\n--- Running with {i} agent(s) ---")
        
        run_df = run_parallel_privmas_evaluation(
            config_path="config.yaml",
            max_rows=max_rows,
            num_agents=i,
            quiet=True,
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

    # Calculate summary for printing
    summary_table = full_results_df.groupby('num_agents').agg(
        avg_e2e_ms=('e2e_ms', 'mean'),
        avg_t_inf_ms=('t_inf_ms', 'mean'),
        avg_delta_sync_ms=('delta_sync_ms', 'mean'),
        avg_c_tax_ms=('c_tax_ms', 'mean'),
        total_runtime_ms=('e2e_ms', 'sum')
    )

    print("\n=== Experiment Summary ===")
    print(summary_table.round(2))

    # Generate detailed report with plots
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'experiment_report')
    
    # We need to structure the data for the report generator
    report_data = []
    for num_agents, group in full_results_df.groupby('num_agents'):
        report_data.append({
            'num_agents': num_agents,
            'e2e_ms': group['e2e_ms'].mean(),
            't_inf_ms': group['t_inf_ms'].mean(),
            'delta_sync_ms': group['delta_sync_ms'].mean(),
            'c_tax_ms': group['c_tax_ms'].mean(),
            'agent_details': group['agent_details'].tolist()
        })

    generate_report(report_data, output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel PrivMAS experiments.")
    parser.add_argument("--max-agents", type=int, default=8, help="Maximum number of agents to test.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for the dataset.")
    
    args = parser.parse_args()
    
    run_experiments(args.max_agents, args.max_rows)

