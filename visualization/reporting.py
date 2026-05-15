"""Generate detailed performance reports and plots for PrivMAS experiments."""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def generate_report(all_results: List[Dict[str, Any]], output_dir: str):
    """
    Generates a comprehensive performance report with plots.

    Args:
        all_results: A list of dictionaries, where each dictionary holds the
                     results from an experiment run (e.g., for a specific agent count).
        output_dir: The directory to save the report and plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Prepare DataFrames ---
    # Main results DataFrame
    summary_df = pd.DataFrame(all_results)

    # Detailed agent-level DataFrame
    agent_details_list = []
    for _, row in summary_df.iterrows():
        num_agents = row['num_agents']
        if isinstance(row['agent_details'], list):
            for agent_detail in row['agent_details']:
                # Ensure agent_detail is a dictionary
                if isinstance(agent_detail, dict):
                    agent_details_list.append({
                        'num_agents': num_agents,
                        'strategy': agent_detail.get('strategy'),
                        'latency_ms': agent_detail.get('latency_ms')
                    })
    agent_df = pd.DataFrame(agent_details_list)

    # --- 2. Generate Plots ---
    plot_latency_distribution(agent_df, output_dir)
    plot_performance_scaling(summary_df, output_dir)
    plot_strategy_distribution(agent_df, output_dir)

    # --- 3. Generate Text Report ---
    with open(os.path.join(output_dir, "performance_report.md"), "w") as f:
        f.write("# PrivMAS Performance Analysis\n\n")
        
        f.write("## Overall Performance Summary\n")
        f.write("This table shows the average performance metrics across all runs for each agent count.\n\n")
        avg_summary = summary_df.groupby('num_agents')[['e2e_ms', 't_inf_ms', 'delta_sync_ms', 'c_tax_ms']].mean()
        f.write(avg_summary.to_markdown())
        f.write("\n\n")

        f.write("## Key Metrics Explained\n")
        f.write("- **e2e_ms**: End-to-end latency. The total time from start to finish.\n")
        f.write("- **t_inf_ms (Critical Path)**: The latency of the slowest agent. This is the theoretical minimum time for the parallel portion.\n")
        f.write("- **delta_sync_ms (Synchronization Delay)**: The time difference between the first and last agent completing their work.\n")
        f.write("- **c_tax_ms (Coordination Tax)**: The overhead from threading, data sharding, and result aggregation (`e2e_ms - t_inf_ms`).\n\n")

        f.write("## Agent Latency Analysis\n")
        f.write("The box plot below shows the latency distribution for each agent strategy. This helps to understand the performance characteristics of each masking approach.\n\n")
        f.write("![Latency Distribution](latency_distribution.png)\n\n")

        f.write("## System Scaling\n")
        f.write("This plot shows how performance metrics change as we increase the number of agents. Ideally, `e2e_ms` should decrease, but `c_tax_ms` may increase due to higher coordination needs.\n\n")
        f.write("![Performance Scaling](performance_scaling.png)\n\n")
        
        f.write("## Strategy Distribution\n")
        f.write("This chart shows the mix of strategies assigned across different agent counts.\n\n")
        f.write("![Strategy Distribution](strategy_distribution.png)\n\n")

    print(f"Report generated in {output_dir}")


def plot_latency_distribution(agent_df: pd.DataFrame, output_dir: str):
    """Plots the distribution of latencies for each agent strategy."""
    if agent_df.empty or 'latency_ms' not in agent_df.columns or 'strategy' not in agent_df.columns:
        return
        
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='strategy', y='latency_ms', data=agent_df)
    plt.title('Agent Latency Distribution by Strategy')
    plt.xlabel('Masking Strategy')
    plt.ylabel('Latency (ms)')
    plt.savefig(os.path.join(output_dir, 'latency_distribution.png'))
    plt.close()


def plot_performance_scaling(summary_df: pd.DataFrame, output_dir: str):
    """Plots how key performance metrics scale with the number of agents."""
    if summary_df.empty:
        return

    plt.figure(figsize=(12, 7))
    
    # Plotting each metric
    sns.lineplot(x='num_agents', y='e2e_ms', data=summary_df, marker='o', label='Total Latency (e2e_ms)')
    sns.lineplot(x='num_agents', y='t_inf_ms', data=summary_df, marker='o', label='Critical Path (t_inf_ms)')
    sns.lineplot(x='num_agents', y='c_tax_ms', data=summary_df, marker='o', label='Coordination Tax (c_tax_ms)')
    
    plt.title('Performance Scaling vs. Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'performance_scaling.png'))
    plt.close()


def plot_strategy_distribution(agent_df: pd.DataFrame, output_dir: str):
    """Plots a pie chart of the strategy distribution for each agent count."""
    if agent_df.empty or 'num_agents' not in agent_df.columns or 'strategy' not in agent_df.columns:
        return

    counts = agent_df.groupby(['num_agents', 'strategy']).size().unstack(fill_value=0)
    
    if counts.empty:
        return

    # Create a subplot for each agent count
    num_agent_configs = len(counts.index)
    fig, axes = plt.subplots(1, num_agent_configs, figsize=(6 * num_agent_configs, 6), squeeze=False)
    
    for i, num_agents in enumerate(counts.index):
        ax = axes[0, i]
        ax.pie(counts.loc[num_agents], labels=counts.columns, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Strategy Distribution for {num_agents} Agents')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_distribution.png'))
    plt.close()
