import pandas as pd
import argparse
from typing import Any, Dict, Optional

from src.single_agent import (
    detect_and_mask_pii_regex, 
    detect_and_mask_pii_spacy,
    detect_and_mask_pii_llm
)

def run_single_agent_evaluation(*, data_path: str = "data/sample_data.csv"):
    """
    Runs the single-agent evaluation on a sample dataset.
    """
    try:
        df = pd.read_csv(data_path)
        print("Original Data:")
        print(df['text'])
        
        # --- Regex Agent ---
        print("\n--- Masking with Regex-based single agent ---")
        df['regex_masked'] = df['text'].apply(detect_and_mask_pii_regex)
        print(df['regex_masked'])

        # --- spaCy Agent ---
        print("\n--- Masking with spaCy-based single agent ---")
        df['spacy_masked'] = df['text'].apply(detect_and_mask_pii_spacy)
        print(df['spacy_masked'])

        # --- LLM Agent (Groq) ---
        print("\n--- Masking with LLM-based single agent (Groq) ---")
        df['llm_masked'] = df['text'].apply(detect_and_mask_pii_llm)
        print(df['llm_masked'])

    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        print("Please ensure you have created the sample data file.")


def run_privmas_evaluation(
    *,
    config_path: str = "config.yaml",
    data_path: str = "data/sample_data.csv",
    max_rows: Optional[int] = None,
    role_counts: Optional[Dict[str, int]] = None,
):
    """Run the PrivMAS multi-agent workflow on the sample dataset.

    Backward compatibility
    ----------------------
    This function is additive. `run_single_agent_evaluation()` remains unchanged.

    Parameters
    ----------
    config_path:
        YAML config path (default: config.yaml).
    max_rows:
        Optional row limit for quick experiments.
    role_counts:
        Optional override for number of agents per role, e.g.
        {"detector": 2, "analyzer": 1, "validator": 1}
    """

    # Import PrivMAS modules lazily so `python main.py` (single-agent)
    # doesn't require installing LangGraph immediately.
    from config import load_config
    from graph.workflow import run_privmas_once
    from evaluation.metrics import compute_all_metrics

    cfg: Dict[str, Any] = load_config(config_path)

    # Apply overrides for ablation studies (vary N)
    if role_counts:
        cfg.setdefault("agents", {})
        cfg["agents"].setdefault("roles", {})
        for role, count in role_counts.items():
            cfg["agents"]["roles"].setdefault(role, {})
            cfg["agents"]["roles"][role]["count"] = int(count)

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found.")
        return

    if max_rows is not None:
        df = df.head(int(max_rows))

    outputs = []

    print("\n=== PrivMAS Evaluation (Multi-Agent) ===")
    for idx, row in df.iterrows():
        text = str(row.get("text", ""))
        label = row.get("label")

        state = run_privmas_once(
            text=text,
            label=None if label is None else str(label),
            run_id=str(idx),
            config=cfg,
        )

        # Compute metrics
        metrics = compute_all_metrics(state)
        state.metrics.update(metrics)

        llm_tokens = metrics.get("llm_tokens", {})
        llm_total_exact = (llm_tokens.get("total_exact") or {}).get("total_tokens")
        llm_total_approx = (llm_tokens.get("total_approx") or {}).get("total_tokens")
        llm_calls = (llm_tokens.get("total_approx") or {}).get("calls")

        outputs.append(
            {
                "id": idx,
                "text": text,
                "label": label,
                "final_strategy": state.final_strategy,
                "final_masked": state.final_masked_text,
                "e2e_ms": (state.timings_ms or {}).get("e2e_ms"),
                "message_count": metrics["coordination"]["message_count"],
                "message_bytes": metrics["coordination"]["message_bytes"],
                "residual_email_count": metrics["privacy_audit"]["residual_email_count"],
                "residual_phone_count": metrics["privacy_audit"]["residual_phone_count"],
                "llm_calls": llm_calls,
                "llm_total_tokens_exact": llm_total_exact,
                "llm_total_tokens_approx": llm_total_approx,
            }
        )

        e2e_ms = outputs[-1]["e2e_ms"]
        e2e_ms_display = float(e2e_ms) if e2e_ms is not None else float("nan")
        print(f"\nRow {idx} | strategy={state.final_strategy} | e2e_ms={e2e_ms_display:.1f}")
        print(state.final_masked_text)

    out_df = pd.DataFrame(outputs)
    print("\n=== PrivMAS Summary ===")
    print(
        out_df[
            [
                "id",
                "final_strategy",
                "e2e_ms",
                "message_count",
                "llm_calls",
                "llm_total_tokens_exact",
                "llm_total_tokens_approx",
                "residual_email_count",
                "residual_phone_count",
            ]
        ]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Agentic_AI: single-agent and PrivMAS runners")
    parser.add_argument("--privmas", action="store_true", help="Run PrivMAS multi-agent workflow")
    parser.add_argument("--config", default="config.yaml", help="Path to PrivMAS config YAML")
    parser.add_argument(
        "--data-path",
        default="data/sample_data.csv",
        help="Path to CSV with a 'text' column (and optional 'label')",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit")
    parser.add_argument("--detectors", type=int, default=None, help="Override detector agent count")
    parser.add_argument("--analyzers", type=int, default=None, help="Override analyzer agent count")
    parser.add_argument("--validators", type=int, default=None, help="Override validator agent count")

    args = parser.parse_args()

    if args.privmas:
        overrides = {}
        if args.detectors is not None:
            overrides["detector"] = args.detectors
        if args.analyzers is not None:
            overrides["analyzer"] = args.analyzers
        if args.validators is not None:
            overrides["validator"] = args.validators

        run_privmas_evaluation(
            config_path=args.config,
            data_path=args.data_path,
            max_rows=args.max_rows,
            role_counts=overrides or None,
        )
    else:
        run_single_agent_evaluation(data_path=args.data_path)


