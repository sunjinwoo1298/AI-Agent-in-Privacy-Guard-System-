import pandas as pd
import argparse
from typing import Any, Dict, Optional


def run_parallel_privmas_evaluation(
    *,
    config_path: str = "config.yaml",
    max_rows: Optional[int] = None,
    num_agents: Optional[int] = None,
    quiet: bool = False,
):
    """Run the parallel PrivMAS workflow on the sample dataset."""
    from config import load_config
    from graph.parallel_workflow import run_parallel_privmas

    cfg: Dict[str, Any] = load_config(config_path)

    if num_agents is not None:
        cfg.setdefault("agents", {})
        cfg["agents"]["generalist_count"] = int(num_agents)

    try:
        df = pd.read_csv('data/sample_data.csv')
    except FileNotFoundError:
        if not quiet:
            print("Error: data/sample_data.csv not found.")
        return pd.DataFrame()

    if max_rows is not None:
        df = df.head(int(max_rows))

    outputs = []

    if not quiet:
        print("\n=== Parallel PrivMAS Evaluation ===")
    for idx, row in df.iterrows():
        text = str(row.get("text", ""))
        label = row.get("label")

        state = run_parallel_privmas(
            text=text,
            label=None if label is None else str(label),
            run_id=str(idx),
            config=cfg,
        )

        outputs.append(
            {
                "id": idx,
                "text": text,
                "label": label,
                "final_strategy": state.final_strategy,
                "final_masked": state.final_masked_text,
                "e2e_ms": (state.timings_ms or {}).get("e2e_ms"),
                "t_inf_ms": (state.timings_ms or {}).get("t_inf_ms"),
                "delta_sync_ms": (state.timings_ms or {}).get("delta_sync_ms"),
                "c_tax_ms": (state.timings_ms or {}).get("c_tax_ms"),
                "errors": len(state.errors),
                "agent_details": state.specialist_results,
            }
        )
        if not quiet:
            e2e_ms = outputs[-1]["e2e_ms"]
            e2e_ms_display = float(e2e_ms) if e2e_ms is not None else float("nan")
            print(f"\nRow {idx} | strategy={state.final_strategy} | e2e_ms={e2e_ms_display:.1f}")
            
            # Display agent-specific details
            if state.specialist_results:
                print("Agent Breakdown:")
                for agent_detail in state.specialist_results:
                    strategy = agent_detail.get('strategy', 'N/A')
                    latency = agent_detail.get('latency_ms', float('nan'))
                    print(f"  - Agent {agent_detail.get('chunk_id')}: {strategy} ({latency:.1f}ms)")

            print(state.final_masked_text)

    out_df = pd.DataFrame(outputs)
    if not quiet:
        print("\n=== Parallel PrivMAS Summary ===")
        print(
            out_df[
                [
                    "id",
                    "final_strategy",
                    "e2e_ms",
                    "t_inf_ms",
                    "delta_sync_ms",
                    "c_tax_ms",
                    "errors",
                ]
            ]
        )
    
    return out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Agentic_AI: Parallel PrivMAS runner")
    parser.add_argument("--config", default="config.yaml", help="Path to PrivMAS config YAML")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit")
    parser.add_argument("--num-agents", type=int, default=None, help="Override generalist agent count for parallel mode")

    args = parser.parse_args()

    run_parallel_privmas_evaluation(
        config_path=args.config,
        max_rows=args.max_rows,
        num_agents=args.num_agents,
    )



