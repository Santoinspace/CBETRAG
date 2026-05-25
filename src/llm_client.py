"""LLMClient abstraction layer — supports AWQ, HuggingFace, and vLLM backends."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math


@dataclass
class LLMResponse:
    text: str
    logprobs: list[float] = field(default_factory=list)  # per-token log probs


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
        ...


class AWQClient(LLMClient):
    """Qwen2.5-7B-Instruct-AWQ via autoawq — primary backend for RTX 4060 8GB."""

    def __init__(self, model_path: str, seed: int = 42):
        import torch
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self.model = AutoAWQForCausalLM.from_quantized(
            model_path, fuse_layers=True, trust_remote_code=False, safetensors=True
        )

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
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

    def __init__(self, model_path: str, device: str = "cuda", seed: int = 42):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device
        ).eval()

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0) -> LLMResponse:
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
        client = VLLMClient(base_url="http://localhost:8000/v1", model="/models/qwen")
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "/models/qwen", api_key: str = "EMPTY",
                 seed: int = 42):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required for VLLMClient: pip install openai")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._seed = seed

    def generate(self, prompt: str, max_new_tokens: int = 512,
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
