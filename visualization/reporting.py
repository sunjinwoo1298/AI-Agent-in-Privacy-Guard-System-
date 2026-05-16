"""Generate detailed performance reports and plots for PrivMAS experiments."""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from collections import defaultdict

def generate_report(all_results: List[Dict[str, Any]], output_dir: str):
    """
    Generates a comprehensive performance and accuracy report with plots.

    Args:
        all_results: A list of dictionaries from experiment runs.
        output_dir: The directory to save the report and plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Prepare DataFrames ---
    summary_df = pd.DataFrame(all_results)
    agent_df = pd.DataFrame([
        {
            'num_agents': row['num_agents'],
            'strategy': ad.get('strategy'),
            'latency_ms': ad.get('latency_ms')
        }
        for _, row in summary_df.iterrows() if isinstance(row.get('agent_details'), list)
        for ad in row['agent_details'] if isinstance(ad, dict)
    ])

    # --- 2. Generate Plots ---
    plot_latency_distribution(agent_df, output_dir)
    plot_performance_scaling(summary_df, output_dir)
    plot_strategy_distribution(agent_df, output_dir)
    plot_accuracy_scaling(summary_df, output_dir)

    # --- 3. Generate Text Report ---
    with open(os.path.join(output_dir, "performance_report.md"), "w") as f:
        f.write("# PrivMAS Performance & Accuracy Analysis\n\n")
        
        f.write("## Overall Performance Summary\n")
        f.write("This table shows average performance metrics for each agent count.\n\n")
        perf_summary = summary_df.groupby('num_agents')[['e2e_ms', 't_inf_ms', 'delta_sync_ms', 'c_tax_ms']].mean()
        f.write(perf_summary.to_markdown(floatfmt=".2f"))
        f.write("\n\n")

        f.write("## Overall Accuracy Summary\n")
        f.write("This table shows average accuracy metrics for each agent count.\n\n")
        acc_summary = summary_df.groupby('num_agents')[['precision', 'recall', 'f1_score']].mean()
        f.write(acc_summary.to_markdown(floatfmt=".2f"))
        f.write("\n\n")

        f.write("### Key Metrics Explained\n")
        f.write("- **e2e_ms**: End-to-end latency.\n")
        f.write("- **t_inf_ms**: Critical Path (slowest agent).\n")
        f.write("- **delta_sync_ms**: Synchronization Delay.\n")
        f.write("- **c_tax_ms**: Coordination Tax (`e2e_ms - t_inf_ms`).\n\n")

        f.write("## System Scaling\n")
        f.write("How performance changes as we add agents.\n\n")
        f.write("![Performance Scaling](performance_scaling.png)\n\n")

        f.write("## Accuracy Scaling\n")
        f.write("How accuracy changes as we add agents.\n\n")
        f.write("![Accuracy Scaling](accuracy_scaling.png)\n\n")

        f.write("## Per-Label Accuracy Analysis\n")
        f.write("Average precision, recall, and F1-score for each PII label, grouped by the number of agents.\n\n")
        f.write(generate_accuracy_tables(summary_df))
        f.write("\n\n")

        f.write("## Agent Latency & Strategy Analysis\n")
        f.write("The plots below show the latency distribution for each strategy and the mix of strategies assigned.\n\n")
        f.write("![Latency Distribution](latency_distribution.png)\n\n")
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
    """Plots the distribution of strategies used."""
    if agent_df.empty or 'strategy' not in agent_df.columns:
        return

    plt.figure(figsize=(10, 6))
    sns.countplot(x='strategy', data=agent_df, hue='strategy', palette='viridis', legend=False)
    plt.title('Masking Strategy Distribution')
    plt.xlabel('Strategy')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'strategy_distribution.png'))
    plt.close()

def plot_accuracy_scaling(summary_df: pd.DataFrame, output_dir: str):
    """Plots how accuracy metrics scale with the number of agents."""
    if summary_df.empty or 'f1_score' not in summary_df.columns:
        return

    plt.figure(figsize=(12, 7))
    
    sns.lineplot(x='num_agents', y='precision', data=summary_df, marker='o', label='Precision')
    sns.lineplot(x='num_agents', y='recall', data=summary_df, marker='o', label='Recall')
    sns.lineplot(x='num_agents', y='f1_score', data=summary_df, marker='o', label='F1-Score')
    
    plt.title('Accuracy Scaling vs. Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_scaling.png'))
    plt.close()

def generate_accuracy_tables(summary_df: pd.DataFrame) -> str:
    """Generates markdown tables for per-label accuracy."""
    if 'accuracy_by_label' not in summary_df.columns:
        return "No per-label accuracy data available.\n"

    all_dfs = []
    for num_agents, group in summary_df.groupby('num_agents'):
        label_accuracies = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0})
        
        for acc_list in group['accuracy_by_label']:
            if isinstance(acc_list, dict):
                per_label = acc_list.get('per_label')
                if isinstance(per_label, dict):
                    for label, metrics in per_label.items():
                        if not isinstance(metrics, dict):
                            continue
                        label_accuracies[label]['tp'] += int(metrics.get('tp', 0))
                        label_accuracies[label]['fp'] += int(metrics.get('fp', 0))
                        label_accuracies[label]['fn'] += int(metrics.get('fn', 0))
                        label_accuracies[label]['count'] += 1
                else:
                    for label, metrics in acc_list.items():
                        if not isinstance(metrics, dict):
                            continue
                        if {'tp', 'fp', 'fn'}.issubset(metrics.keys()):
                            label_accuracies[label]['tp'] += int(metrics.get('tp', 0))
                            label_accuracies[label]['fp'] += int(metrics.get('fp', 0))
                            label_accuracies[label]['fn'] += int(metrics.get('fn', 0))
                            label_accuracies[label]['count'] += 1
            elif isinstance(acc_list, list):
                for item in acc_list:
                    if not isinstance(item, dict):
                        continue
                    per_label = item.get('per_label')
                    if isinstance(per_label, dict):
                        for label, metrics in per_label.items():
                            if not isinstance(metrics, dict):
                                continue
                            label_accuracies[label]['tp'] += int(metrics.get('tp', 0))
                            label_accuracies[label]['fp'] += int(metrics.get('fp', 0))
                            label_accuracies[label]['fn'] += int(metrics.get('fn', 0))
                            label_accuracies[label]['count'] += 1
                    else:
                        for label, metrics in item.items():
                            if not isinstance(metrics, dict):
                                continue
                            if {'tp', 'fp', 'fn'}.issubset(metrics.keys()):
                                label_accuracies[label]['tp'] += int(metrics.get('tp', 0))
                                label_accuracies[label]['fp'] += int(metrics.get('fp', 0))
                                label_accuracies[label]['fn'] += int(metrics.get('fn', 0))
                                label_accuracies[label]['count'] += 1

        table_data = []
        for label, metrics in sorted(label_accuracies.items()):
            precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
            recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            table_data.append({
                'Label': label,
                'Precision': f"{precision:.2f}",
                'Recall': f"{recall:.2f}",
                'F1-Score': f"{f1:.2f}",
            })
        
        if table_data:
            df = pd.DataFrame(table_data)
            all_dfs.append(f"#### {num_agents} Agent(s)\n" + df.to_markdown(index=False) + "\n")

    return "\n".join(all_dfs)
    
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
