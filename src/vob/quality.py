"""FP8 correctness gate (Phase 4.5): quality delta of each precision vs BF16.

Throughput parity proves we time correctly, NOT that FP8 generates as well as
BF16. Before the full sweep we run a fixed held-out prompt set through BF16 and
each FP8 variant and record a cheap quality proxy per precision level:

  - perplexity on a fixed text set (prompt_logprobs from the vLLM offline API), and
  - optional exact-match accuracy on a small GSM8K/MMLU subset.

Runs INSIDE a SLURM GPU allocation. Uses vLLM's offline `LLM` API (not the
server) so we get per-token logprobs directly. A gross degradation flags that
precision level so its "FP8 is faster" claim is reported with the quality cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class QualityResult:
    precision: str
    model_id: str
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    n_prompts: int = 0
    n_tasks: int = 0
    perplexity_delta_pct: Optional[float] = None   # vs BF16, filled by compare()
    accuracy_delta_pts: Optional[float] = None
    flagged: bool = False
    notes: list[str] = field(default_factory=list)


def compute_perplexity(model_id: str, texts: list[str], *,
                       quantization: Optional[str] = None,
                       kv_cache_dtype: str = "auto",
                       max_len: int = 2048) -> float:
    """Mean per-token perplexity over `texts` using vLLM prompt_logprobs.

    Imports vLLM lazily so this module imports fine on CPU-only hosts.
    """
    from vllm import LLM, SamplingParams  # lazy: GPU-only

    kwargs: dict[str, Any] = {"model": model_id, "max_model_len": max_len, "enforce_eager": True}
    if quantization:
        kwargs["quantization"] = quantization
    if kv_cache_dtype != "auto":
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    llm = LLM(**kwargs)

    # prompt_logprobs=1 returns the logprob vLLM assigned to each actual prompt token.
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)
    outputs = llm.generate(texts, sp)

    total_logprob, total_tokens = 0.0, 0
    for out in outputs:
        for tok_lp in (out.prompt_logprobs or []):
            if not tok_lp:
                continue
            # tok_lp maps token_id -> Logprob; take the actual (first) entry.
            lp = next(iter(tok_lp.values()))
            logprob = getattr(lp, "logprob", None)
            if logprob is not None and not math.isinf(logprob):
                total_logprob += logprob
                total_tokens += 1
    if total_tokens == 0:
        return float("nan")
    return math.exp(-total_logprob / total_tokens)


def evaluate(precision: str, model_id: str, texts: list[str], *,
             quantization: Optional[str] = None, kv_cache_dtype: str = "auto") -> QualityResult:
    ppl = compute_perplexity(model_id, texts, quantization=quantization, kv_cache_dtype=kv_cache_dtype)
    return QualityResult(precision=precision, model_id=model_id,
                         perplexity=ppl, n_prompts=len(texts))


def compare(results: list[QualityResult], *, baseline_precision: str = "bf16",
            ppl_flag_pct: float = 5.0) -> list[QualityResult]:
    """Fill deltas vs the baseline precision and flag gross degradation.

    A precision level is flagged when its perplexity is >`ppl_flag_pct`% worse
    than BF16 — such a level is excluded from fair "FP8 is faster" claims unless
    the cost is disclosed.
    """
    base = next((r for r in results if r.precision == baseline_precision), None)
    if base is None or not base.perplexity:
        return results
    for r in results:
        if r.perplexity and base.perplexity:
            r.perplexity_delta_pct = round((r.perplexity - base.perplexity) / base.perplexity * 100, 2)
            if r.precision != baseline_precision and r.perplexity_delta_pct > ppl_flag_pct:
                r.flagged = True
                r.notes.append(f"perplexity {r.perplexity_delta_pct:+.1f}% vs BF16 (> {ppl_flag_pct}% gate)")
    return results
