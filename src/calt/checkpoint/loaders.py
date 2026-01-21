from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_inference_bundle(run_dir: str, device="cpu"):
    model = AutoModelForSeq2SeqLM.from_pretrained(f"{run_dir}/model").to(device)
    tok = AutoTokenizer.from_pretrained(f"{run_dir}/tokenizer", use_fast=True)
    return model, tok
