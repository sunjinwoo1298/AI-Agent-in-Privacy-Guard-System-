# Agent Optimization in Privacy Guard System

This repo starts with a **single-agent baseline** for PII masking and evolves into **PrivMAS** (Privacy-aware Multi-Agent System) for experiments on coordination overhead vs performance.

## Project Layout
- [main.py](main.py): Runner (single-agent baseline + PrivMAS runner)
- [src/single_agent.py](src/single_agent.py): Existing masking functions (regex / spaCy / Groq LLM)
- [config.yaml](config.yaml): PrivMAS experiment configuration (agent counts, strategies, aggregation)
- [config.py](config.py): Loads config + env vars (also provides `GROQ_API_KEY` for `single_agent.py`)
- [privacy/masker.py](privacy/masker.py): `PrivacyMasker` heuristic router that reuses existing functions
- [agents/supervisor.py](agents/supervisor.py): Supervisor planning + aggregation logic
- [graph/workflow.py](graph/workflow.py): LangGraph workflow definition
- [evaluation/metrics.py](evaluation/metrics.py): Latency + coordination overhead + residual PII audit metrics

## Setup
1. Install dependencies:
	- `pip install -r requirements.txt`
2. (Optional) enable Groq LLM masking:
	- Copy `.env.example` to `.env`
	- Set `GROQ_API_KEY` in `.env`

## Run: Single-Agent Baseline (backward compatible)
- `python main.py`

This loads `data/sample_data.csv` and prints masked outputs for:
- regex-only
- spaCy NER
- Groq LLM

## Run: PrivMAS (Multi-Agent)
- `python main.py --privmas`

Optional ablation overrides (vary N):
- `python main.py --privmas --detectors 2 --analyzers 1 --validators 1`

Config file:
- Edit `config.yaml` to change default strategies, aggregation policy, and role counts.