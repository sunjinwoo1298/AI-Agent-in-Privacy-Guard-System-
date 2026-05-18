"""Hugging Face-backed agent implementation."""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Optional, Tuple

import torch

from mas_privacy_eval.agents.prompts import (
    ANALYZER_PROMPT,
    CONSENSUS_PROMPT,
    DETECTOR_PROMPT,
    VALIDATOR_PROMPT,
)
from mas_privacy_eval.agents.types import AgentOutput, AgentParseError
from mas_privacy_eval.config import ModelConfig


logger = logging.getLogger(__name__)


class RealLLMAgent:
    """A role-specialized agent backed by a local Hugging Face model."""

    ROLE_PROMPTS = {
        "Detector": DETECTOR_PROMPT,
        "Analyzer": ANALYZER_PROMPT,
        "Validator": VALIDATOR_PROMPT,
        "Consensus": CONSENSUS_PROMPT,
    }

    MAX_RETRIES = 2

    def __init__(
        self,
        *,
        agent_id: str,
        role: str,
        hf_model,
        hf_tokenizer,
        model_cfg: ModelConfig,
        verbose: bool = False,
    ) -> None:
        if role not in self.ROLE_PROMPTS:
            raise ValueError(f"Unknown role: {role}")
        self.agent_id = agent_id
        self.role = role
        self._model = hf_model
        self._tokenizer = hf_tokenizer
        self._verbose = verbose
        self._system = self.ROLE_PROMPTS[role]
        self._cfg = model_cfg

    def _call_model(self, user_message: str) -> Tuple[str, float, int, int]:
        """One HF generation call returning (text, latency_ms, in_tokens, out_tokens)."""

        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": user_message},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self._tokenizer(prompt, return_tensors="pt")
        model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=self._cfg.max_new_tokens,
                do_sample=self._cfg.do_sample,
                temperature=self._cfg.temperature,
                top_p=self._cfg.top_p,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        input_length = int(model_inputs["input_ids"].shape[1])
        generated_text_ids = generated_ids[0, input_length:]
        text = self._tokenizer.decode(generated_text_ids, skip_special_tokens=True)

        in_tok = input_length
        out_tok = int(generated_text_ids.shape[0])
        return text.strip(), latency_ms, in_tok, out_tok

    @staticmethod
    def _parse_json(text: str) -> Dict:
        cleaned = text.strip()
        for fence in ("```json", "```JSON", "```"):
            cleaned = cleaned.replace(fence, "")
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise AgentParseError(f"Cannot parse JSON from: {cleaned[:200]!r}")

    def infer(self, text: str, prior_context: str = "") -> AgentOutput:
        """Run this agent on a sample text with optional prior context."""

        if prior_context:
            user_msg = f"PRIOR AGENT ANALYSES:\n{prior_context}\n\nTEXT TO EVALUATE:\n{text}"
        else:
            user_msg = f"TEXT TO EVALUATE:\n{text}"

        raw = ""
        latency_ms = 0.0
        in_tok = 0
        out_tok = 0
        parsed: Optional[Dict] = None
        retries = 0
        error_msg: Optional[str] = None
        parse_ok = False

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                raw, lat, in_t, out_t = self._call_model(user_msg)
                latency_ms += lat
                in_tok += in_t
                out_tok += out_t
                parsed = self._parse_json(raw)
                parse_ok = True
                break
            except AgentParseError as exc:
                retries += 1
                error_msg = str(exc)
                user_msg += "\n\nCRITICAL: Respond ONLY with a valid JSON object. No other text."
            except Exception as exc:  # noqa: BLE001
                retries += 1
                error_msg = f"Inference error: {exc}"
                time.sleep(0.5)

        label: Optional[int] = None
        confidence = 0.5
        reasoning = ""
        if parsed is not None:
            try:
                raw_label = parsed.get("label")
                label = int(raw_label) if raw_label in (0, 1, "0", "1") else None
                confidence = float(parsed.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                reasoning = str(parsed.get("reasoning", ""))
            except (TypeError, ValueError) as exc:
                error_msg = f"Field extraction error: {exc}"

        if self._verbose:
            status = "OK" if parse_ok else "FAIL"
            logger.info(
                "Agent %s (%s): %s label=%s conf=%.2f lat=%.0fms tok=%d+%d",
                self.agent_id,
                self.role,
                status,
                label,
                confidence,
                latency_ms,
                in_tok,
                out_tok,
            )

        return AgentOutput(
            agent_id=self.agent_id,
            role=self.role,
            raw_response=raw,
            parsed=parsed,
            label=label,
            confidence=confidence,
            reasoning=reasoning,
            latency_ms=round(latency_ms, 2),
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            parse_success=parse_ok,
            parse_retries=retries,
            error_message=error_msg,
            timestamp=time.time(),
        )
