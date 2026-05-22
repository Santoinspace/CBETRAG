"""Task 0: Environment validation script.
Success criterion: runs without error, last line prints [ENV OK]
"""
import os, sys


def check_venv():
    assert ".venv" in sys.prefix, f"Virtual env not active: {sys.prefix}"
    print(f"[VENV] {sys.prefix}")


def check_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[CUDA] {torch.cuda.get_device_name(0)}, VRAM={vram_gb:.1f}GB")
    assert vram_gb >= 7, f"VRAM {vram_gb:.1f}GB < 7GB required"


def check_adaragUE():
    base = os.path.join(os.path.dirname(__file__), "..", "AdaRAGUE")
    for d in ["data", "standard_retriever"]:
        assert os.path.isdir(os.path.join(base, d)), f"Missing AdaRAGUE/{d}"
    print("[AdaRAGUE] clone OK")


def check_nli():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "nli-deberta-v3-base")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda().eval()
    vram_before = torch.cuda.memory_allocated() / 1e9
    inputs = tok("The sky is blue.", "The sky has color.", return_tensors="pt",
                 truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    vram_nli = torch.cuda.memory_allocated() / 1e9
    labels = ["contradiction", "entailment", "neutral"]
    pred = labels[logits.argmax().item()]
    print(f"[NLI] {pred} | VRAM used: {vram_nli:.2f}GB")
    assert vram_nli < 0.6, f"NLI VRAM {vram_nli:.2f}GB >= 0.6GB"
    return model  # keep loaded for joint check


def check_llm():
    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-7B-Instruct-AWQ")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoAWQForCausalLM.from_quantized(
        model_path, fuse_layers=True, trust_remote_code=False, safetensors=True
    )
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[LLM] AWQ loaded | VRAM used: {vram_gb:.2f}GB")
    assert vram_gb < 5.5, f"LLM VRAM {vram_gb:.2f}GB >= 5.5GB"

    inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"[LLM] output: '{text}'")
    return model


if __name__ == "__main__":
    check_venv()
    check_cuda()
    check_adaragUE()
    nli_model = check_nli()
    llm_model = check_llm()

    import torch
    total_vram = torch.cuda.memory_allocated() / 1e9
    assert total_vram < 7.5, f"Total VRAM {total_vram:.1f}GB >= 7.5GB"
    print(f"[ENV OK] Total VRAM: {total_vram:.1f}GB / 8.0GB")
