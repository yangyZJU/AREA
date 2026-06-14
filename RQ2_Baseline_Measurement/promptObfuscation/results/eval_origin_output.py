import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--csv_path", type=str, default="./obfuscate_truthQA/obf_sys_output.csv")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()

def ensure_columns(df: pd.DataFrame) -> None:
    needed = ["original_system_prompt", "test_user_prompt"]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

def apply_chat(tokenizer, messages, device):
    """Build model input ids using chat template if available."""
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
    except Exception:
        sys = next((m["content"] for m in messages if m["role"] == "system"), "")
        usr = next((m["content"] for m in messages if m["role"] == "user"), "")
        prompt = f"<|system|>\n{sys}\n<|user|>\n{usr}\n<|assistant|>\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return input_ids

@torch.no_grad()
def gen_once(model, tokenizer, messages, device, gen_cfg):
    input_ids = apply_chat(tokenizer, messages, device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=gen_cfg["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id)
    )
    text = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    return text.strip()

def main():
    args = parse_args()

    # Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype)
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    gen_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p
    }

    # Read CSV
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(args.csv_path)
    df = pd.read_csv(args.csv_path)
    ensure_columns(df)

    df["origin_output"] = ""
    df["origin_output2"] = ""
    df["output_wo_sys"] = ""

    # Generate per row
    cnt = 1
    for i, row in df.iterrows():
        print(cnt)
        cnt += 1
        sys_prompt = str(row["original_system_prompt"])
        user_prompt = str(row["test_user_prompt"])

        # A) origin_output (system + user)
        messages_with = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        origin_output = gen_once(model, tokenizer, messages_with, args.device, gen_cfg)

        origin_output2 = gen_once(model, tokenizer, messages_with, args.device, gen_cfg)

        # C) output_wo_sys (user only)
        messages_wo = [{"role": "user", "content": user_prompt}]
        output_wo_sys = gen_once(model, tokenizer, messages_wo, args.device, gen_cfg)

        df.at[i, "origin_output"] = origin_output
        df.at[i, "origin_output2"] = origin_output2
        df.at[i, "output_wo_sys"] = output_wo_sys

        if (i + 1) % 10 == 0:
            df.to_csv(args.csv_path, index=False, encoding="utf-8-sig")
            print(f"[Progress] processed {i+1}/{len(df)}")

    df.to_csv(args.csv_path, index=False, encoding="utf-8-sig")
    print(f"Done. Saved back to: {args.csv_path}")

if __name__ == "__main__":
    main()
