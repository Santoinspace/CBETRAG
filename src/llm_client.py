"""LLMClient abstraction layer — supports AWQ, HuggingFace, and vLLM backends.

Includes MD5-based disk cache for deterministic LLM calls (DAG extraction,
claim extraction, parametric probe). Final answer generation should pass
use_cache=False since context changes across iterations.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import logging
import math
import time

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    logprobs: list[float] = field(default_factory=list)  # per-token log probs


class LLMClient(ABC):
    """Base LLM client with optional disk cache."""

    def __init__(self, cache_dir: str = "./.llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_hits = 0
        self._cache_misses = 0

    def _cache_key(self, prompt: str, max_new_tokens: int,
                   temperature: float) -> str:
        """MD5 of model+params+content — prevents cross-model cache contamination."""
        model_name = getattr(self, 'model_name', 'unknown')
        content = (
            f"model={model_name}|||"
            f"temp={temperature}|||"
            f"max_tokens={max_new_tokens}|||"
            f"prompt={prompt}"
        )
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.0, use_cache: bool = True) -> LLMResponse:
        """Generate with disk cache + retry on empty response.

        Retries up to 3 times with exponential backoff (1s, 2s, 4s) when
        the LLM returns an empty response (e.g. vLLM concurrent timeout).
        Empty responses are never cached.
        """
        if not use_cache:
            return self._generate_with_retry(prompt, max_new_tokens, temperature)

        key = self._cache_key(prompt, max_new_tokens, temperature)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            self._cache_hits += 1
            try:
                data = json.loads(cache_file.read_text(encoding='utf-8'))
                text = data.get("text", "")
                # Skip cached empty responses — they are stale/invalid
                if text and text.strip():
                    return LLMResponse(
                        text=text,
                        logprobs=data.get("logprobs", [])
                    )
                # Empty cached entry — fall through to retry
                logger.debug("skipping cached empty response for key %s", key[:16])
            except Exception:
                # Corrupted cache entry — fall through to actual call
                pass

        self._cache_misses += 1
        response = self._generate_with_retry(prompt, max_new_tokens, temperature)

        # Never cache empty responses
        if not response.text.strip():
            logger.warning("LLM returned empty response, not caching (key=%s)", key[:16])
            return response

        try:
            cache_file.write_text(
                json.dumps({
                    "text": response.text,
                    "logprobs": response.logprobs,
                    "prompt_preview": prompt[:200],
                }, ensure_ascii=False),
                encoding='utf-8',
            )
        except Exception as e:
            logger.debug("cache write failed: %s", e)

        return response

    def _generate_with_retry(self, prompt: str, max_new_tokens: int = 512,
                             temperature: float = 0.0,
                             max_retries: int = 3) -> LLMResponse:
        """Call _generate with exponential backoff retry on empty response."""
        last_response = LLMResponse(text="", logprobs=[])
        for attempt in range(max_retries + 1):
            try:
                response = self._generate(prompt, max_new_tokens, temperature)
            except Exception as e:
                logger.warning("LLM call failed (attempt %d/%d): %s",
                               attempt + 1, max_retries + 1, e)
                response = LLMResponse(text="", logprobs=[])

            if response.text.strip():
                if attempt > 0:
                    logger.info("LLM call succeeded on retry %d", attempt + 1)
                return response

            last_response = response
            if attempt < max_retries:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning("LLM returned empty response (attempt %d/%d), "
                               "retrying in %ds...", attempt + 1, max_retries + 1, wait)
                time.sleep(wait)

        logger.error("LLM returned empty response after %d attempts", max_retries + 1)
        return last_response

    @abstractmethod
    def _generate(self, prompt: str, max_new_tokens: int = 512,
                  temperature: float = 0.0) -> LLMResponse:
        """Subclass implementation — called when cache misses."""
        ...

    def cache_stats(self) -> str:
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return "No LLM calls yet"
        rate = self._cache_hits / total * 100
        return (f"LLM cache: {self._cache_hits}/{total} "
                f"hits ({rate:.0f}%)")


class AWQClient(LLMClient):
    """Qwen2.5-7B-Instruct-AWQ via autoawq — primary backend for RTX 4060 8GB."""

    def __init__(self, model_path: str, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        import torch
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        torch.manual_seed(seed)
        self.model_name = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self.model = AutoAWQForCausalLM.from_quantized(
            model_path, fuse_layers=True, trust_remote_code=False, safetensors=True
        )

    def _generate(self, prompt: str, max_new_tokens: int = 512,
                  temperature: float = 0.0) -> LLMResponse:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        logprobs = []
        for step, scores in enumerate(out.scores):
            if step >= len(gen_ids):
                break
            lp = torch.log_softmax(scores[0], dim=-1)
            logprobs.append(lp[gen_ids[step]].item())
        return LLMResponse(text=text, logprobs=logprobs)


class HFClient(LLMClient):
    """Standard HuggingFace backend (bfloat16) — for environments with more VRAM."""

    def __init__(self, model_path: str, device: str = "cuda", seed: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.manual_seed(seed)
        self.model_name = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        ).eval()

    def _generate(self, prompt: str, max_new_tokens: int = 512,
                  temperature: float = 0.0) -> LLMResponse:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        logprobs = [
            torch.log_softmax(scores[0], dim=-1)[gen_ids[i]].item()
            for i, scores in enumerate(out.scores) if i < len(gen_ids)
        ]
        return LLMResponse(text=text, logprobs=logprobs)


class VLLMClient(LLMClient):
    """OpenAI-compatible vLLM backend — connects to a running vLLM server.

    Usage:
        client = VLLMClient(base_url="http://localhost:8000/v1", model="qwen25-7b")
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "/models/qwen", api_key: str = "EMPTY",
                 seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for VLLMClient: pip install openai")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._seed = seed
        self.model_name = model
        self.base_url = base_url

    def _generate(self, prompt: str, max_new_tokens: int = 512,
                  temperature: float = 0.0) -> LLMResponse:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                seed=self._seed,
                logprobs=True,
                top_logprobs=1,
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            # Extract per-token logprobs from vLLM response
            logprobs: list[float] = []
            if choice.logprobs and choice.logprobs.content:
                for token_logprob in choice.logprobs.content:
                    lp = token_logprob.logprob  # log-prob of the chosen token
                    logprobs.append(lp)
            return LLMResponse(text=text.strip(), logprobs=logprobs)
        except Exception:
            # Fallback: return empty response on failure
            return LLMResponse(text="", logprobs=[])


def build_client(backend: str, model_path: str, **kwargs) -> LLMClient:
    """Factory. backend: 'awq' (default for 4060), 'hf', or 'vllm'."""
    if backend == "hf":
        return HFClient(model_path, **kwargs)
    if backend == "vllm":
        return VLLMClient(**kwargs)
    return AWQClient(model_path, **kwargs)
