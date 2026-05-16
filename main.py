import pandas as pd
import argparse
import json
from typing import Any, Dict, List, Optional


def run_parallel_privmas_evaluation(
    *,
    config_path: str = "config.yaml",
    data_path: str = "data/test_data.json",
    max_rows: Optional[int] = None,
    num_agents: Optional[int] = None,
    quiet: bool = False,
    test_data: Optional[List[Dict[str, Any]]] = None,
):
    """Run the parallel PrivMAS workflow on the sample dataset."""
    from config import load_config
    from graph.parallel_workflow import run_in_parallel
    from evaluation.accuracy import evaluate_pii_detection

    cfg: Dict[str, Any] = load_config(config_path)

    if num_agents is not None:
        cfg.setdefault("agents", {})
        cfg["agents"]["generalist_count"] = int(num_agents)

    if test_data is None:
        try:
            with open(data_path, 'r') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            if not quiet:
                print(f"Error: {data_path} not found.")
            return pd.DataFrame()

    if max_rows is not None:
        test_data = test_data[:int(max_rows)]

    df = pd.DataFrame(test_data)

    outputs = []

    if not quiet:
        print("\n=== Parallel PrivMAS Evaluation ===")
    for idx, row in df.iterrows():
        text = str(row.get("text", ""))
        label = row.get("label")
        ground_truth_entities = row.get("entities", [])

        state = run_in_parallel(
            text=text,
            label=None if label is None else str(label),
            run_id=str(idx),
            config=cfg,
        )

        # Evaluate accuracy
        accuracy_results = evaluate_pii_detection(
            predicted_entities=state.aggregated_entities,
            ground_truth_entities=ground_truth_entities
        )
        overall_metrics = {
            "overall_precision": accuracy_results["precision"],
            "overall_recall": accuracy_results["recall"],
            "overall_f1_score": accuracy_results["f1"],
            "tp": accuracy_results["tp"],
            "fp": accuracy_results["fp"],
            "fn": accuracy_results["fn"],
        }


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
                "precision": overall_metrics["overall_precision"],
                "recall": overall_metrics["overall_recall"],
                "f1_score": overall_metrics["overall_f1_score"],
                "accuracy_by_label": accuracy_results,
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
            print(f"Accuracy: Precision={overall_metrics['overall_precision']:.2f}, Recall={overall_metrics['overall_recall']:.2f}, F1={overall_metrics['overall_f1_score']:.2f}")


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
                    "precision",
                    "recall",
                    "f1_score",
                ]
            ]
        )
    
    return out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Agentic_AI: Parallel PrivMAS runner")
    parser.add_argument("--config", default="config.yaml", help="Path to PrivMAS config YAML")
    parser.add_argument(
        "--data-path",
        default="data/sample_data.csv",
        help="Path to CSV with a 'text' column (and optional 'label')",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit")
    parser.add_argument("--num-agents", type=int, default=None, help="Override generalist agent count for parallel mode")

    args = parser.parse_args()

    run_parallel_privmas_evaluation(
        config_path=args.config,
        max_rows=args.max_rows,
        num_agents=args.num_agents,
    )



