# test.py
'''
CUDA_VISIBLE_DEVICES=1 nohup python -u test_soft_prompt.py \
  --run-dir "llama3_checkpoint/id_1" \
  --prompts-json "./data/attack/llama3_attack_prompts.json" \
  --compare-original \
  --inference-alpha 1 \
  --csv-out "test_attack.csv" \
  > test20/id_1/attack.log 2>&1 &

'''
import warnings
warnings.filterwarnings("ignore")
import time
import os, json, argparse
from typing import List, Tuple, Optional
import pandas as pd
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()

# Optional progress bar.
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def find_subsequence(full: Tensor, sub: Tensor, tokenizer=None, tolerate: int = 1) -> Tuple[int, int]:
    """Find sub inside full and return (start, end); raise an error if no match exists."""
    full_ids = full[0]; sub_ids = sub[0]
    F, S = full_ids.size(0), sub_ids.size(0)

    # 1) exact
    for i in range(F - S + 1):
        if torch.equal(full_ids[i:i+S], sub_ids):
            return i, i + S

    # 2) relaxed by tolerate
    for i in range(F - S + 1):
        mism = (full_ids[i:i+S] != sub_ids).sum().item()
        if mism <= tolerate:
            return i, i + S

    # 3) decode fallback
    if tokenizer is not None:
        sub_text = tokenizer.decode(sub_ids).strip()
        for i in range(F):
            for j in range(i+1, min(i + S + 4, F + 1)):
                cand = tokenizer.decode(full_ids[i:j]).strip()
                if cand == sub_text:
                    return i, j

    raise ValueError("Subsequence not found in full sequence")


class BenignHarness:
    """
    System + Defense + {SOFT} + User
    Supports baseline generation without the soft prompt, generation with the
    soft prompt inserted at the embedding level, and original generation without
    either the defense prompt or the soft prompt.
    """
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype_str: str,
        system_prompt: str,
        defense_prompt: str,
        user_prompts: List[str],
        gen_max_new: int = 512,
        inference_alpha: float = 0.4,
        attention_type: str = "input_last_token",  # Kept as a compatibility placeholder.
    ):
        self.system_prompt = system_prompt
        self.defense_prompt = defense_prompt
        self.user_prompts = user_prompts
        self.gen_max_new = int(gen_max_new)
        self.inference_alpha = float(inference_alpha)

        dtype = getattr(torch, dtype_str)
        model_kwargs = {"torch_dtype": dtype}
        if str(device).startswith("cuda"):
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device
        self.emb_layer = self.model.get_input_embeddings()
        self.model_input_device = self.emb_layer.weight.device

        self._prepare_static_parts()

    def _prepare_static_parts(self):
        tk = self.tokenizer
        soft_placeholder = "{SOFT}"

        # Build the template only to locate defense_slice.
        temp = tk.apply_chat_template(
            [
                {"role": "system", "content": f"{self.system_prompt} {self.defense_prompt} {soft_placeholder}"},
                {"role": "user", "content": "PLACEHOLDER"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if tk.bos_token and temp.startswith(tk.bos_token):
            temp = temp.replace(tk.bos_token, "", 1)
        before_str_baseline, _ = temp.split(soft_placeholder)
        before_ids_baseline = tk([before_str_baseline], padding=False, return_tensors="pt")["input_ids"].to(self.model_input_device)

        # Locate the defense prompt token span inside the prefix.
        defense_ids = tk(self.defense_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model_input_device)
        dstart, dend = find_subsequence(before_ids_baseline, defense_ids, tokenizer=tk, tolerate=1)
        self.defense_slice = slice(dstart, dend)  # Kept for consistency; not used directly in this test.

        # Cache the before/after embeddings for each user prompt to speed up generation.
        self.before_embeds_list = []
        self.after_embeds_list = []

        for up in self.user_prompts:
            messages = [
                {"role": "system", "content": f"{self.system_prompt} {self.defense_prompt} {soft_placeholder}"},
                {"role": "user", "content": up},
            ]
            s = tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if tk.bos_token and s.startswith(tk.bos_token):
                s = s.replace(tk.bos_token, "", 1)
            before_str, after_str = s.split(soft_placeholder)
            before_ids = tk([before_str], padding=False, return_tensors="pt")["input_ids"].to(self.model_input_device)
            after_ids = tk([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model_input_device)
            self.before_embeds_list.append(self.emb_layer(before_ids))
            self.after_embeds_list.append(self.emb_layer(after_ids))

    @torch.no_grad()
    def generate_with_soft_prompt(self, soft_prompt: Tensor, idx: int) -> str:
        be = self.before_embeds_list[idx]
        ae = self.after_embeds_list[idx]
        inp = torch.cat([be, soft_prompt * self.inference_alpha, ae], dim=1)
        attn_mask = torch.ones(inp.shape[:-1], dtype=torch.long, device=inp.device)
        out_ids = self.model.generate(
            inputs_embeds=inp,
            attention_mask=attn_mask,
            max_new_tokens=self.gen_max_new,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
            #do_sample=False
        )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    @torch.no_grad()
    def generate_baseline(self, idx: int) -> str:
        """Generate the defended baseline without the soft prompt."""
        messages = [
            {"role": "system", "content": f"{self.system_prompt} {self.defense_prompt}"},
            {"role": "user", "content": self.user_prompts[idx]},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model_input_device)
        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.gen_max_new,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
            #do_sample=False
        )
        return self.tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    @torch.no_grad()
    def generate_original(self, idx: int) -> str:
        """Generate the original output with only system_prompt and user_prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompts[idx]},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model_input_device)
        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.gen_max_new,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
            #do_sample=False
        )
        return self.tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def load_run_dir(run_dir: str, fallback_system: Optional[str], fallback_defense: Optional[str]):
    """
    Loads soft_prompt.pt and optimized_soft_prompt.json from a training output directory.
    CLI values are used as fallbacks when metadata is missing.
    """
    ckpt = os.path.join(run_dir, "soft_prompt.pt")
    meta = os.path.join(run_dir, "optimized_soft_prompt.json")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"soft_prompt.pt not found under: {run_dir}")

    soft_pkg = torch.load(ckpt, map_location="cpu")
    if "soft_prompt" not in soft_pkg:
        raise ValueError(f"'soft_prompt' not found in {ckpt}")
    soft_prompt = soft_pkg["soft_prompt"]

    system_prompt = fallback_system
    defense_prompt = fallback_defense

    if os.path.exists(meta):
        with open(meta, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
        system_prompt = meta_json.get("system_prompt", system_prompt)
        defense_prompt = (
            meta_json.get("defensive_instruction")
            or meta_json.get("defense_prompt")
            or defense_prompt
        )

    if not system_prompt or not defense_prompt:
        raise ValueError("system_prompt / defense_prompt not provided (neither meta nor CLI).")

    return soft_prompt, system_prompt, defense_prompt


def parse_args():
    ap = argparse.ArgumentParser(description="Benign prompts sanity test with/without soft prompt.")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16","bfloat16","float32"])

    # Training output directory containing soft_prompt.pt and optimized_soft_prompt.json.
    ap.add_argument("--run-dir", type=str, default="results_soft/sys")

    # Optional CLI fallbacks when metadata is missing.
    ap.add_argument("--system-prompt", type=str, default=None)
    ap.add_argument("--defense-prompt", type=str, default=None)

    # Benign prompts; accepts either source or both.
    ap.add_argument("--prompts-json", type=str, default=None, help="JSON file: list[str]")
    ap.add_argument("--prompt", action="append", default=None, help="Provide one or more prompts via CLI; can repeat")
    ap.add_argument("--prompt-csv", type=str, default=None, help="Provide csv")


    # Generation settings.
    ap.add_argument("--gen-max-new", type=int, default=256)
    ap.add_argument("--inference-alpha", type=float, default=1)
    ap.add_argument("--compare-baseline", action="store_true")
    ap.add_argument("--compare-original", action="store_true", help="Generate original output (no defense, no soft)")

    # Output paths.
    ap.add_argument("--output", type=str, default=None, help="output json path; default: RUN_DIR/test_outputs.json")
    ap.add_argument("--csv-out", type=str, default=None, help="(CSV) output path; default: RUN_DIR/test_outputs.csv")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Load the soft prompt and system/defense prompts.
    soft_prompt, system_prompt, defense_prompt = load_run_dir(
        run_dir=args.run_dir,
        fallback_system=args.system_prompt,
        fallback_defense=args.defense_prompt,
    )

    # 2) Collect benign prompts.
    benign_prompts: List[str] = []
    if args.prompts_json:
        if not os.path.exists(args.prompts_json):
            raise FileNotFoundError(args.prompts_json)
        with open(args.prompts_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert isinstance(loaded, list) and all(isinstance(x, str) for x in loaded), \
            "--prompts-json must be JSON list[str]"
        benign_prompts.extend(loaded)
    if args.prompt:
        for p in args.prompt:
            if isinstance(p, str) and p.strip():
                benign_prompts.append(p.strip())
    if args.prompt_csv:
        df = pd.read_csv(args.prompt_csv)
        prompt_col = "test_user_prompt"
        if prompt_col not in df.columns:
            raise ValueError(
                f"--prompt-csv must contain a '{prompt_col}' column. "
                f"Available columns: {list(df.columns)}"
            )
        csv_prompts = (
            df[prompt_col]
            .dropna()
            .astype(str)
            .map(str.strip)
            .tolist()
        )
        benign_prompts.extend([p for p in csv_prompts if p])
    if len(benign_prompts) == 0:
        raise ValueError("No benign prompts provided. Use --prompts-json and/or --prompt.")

    # 3) Build the evaluation harness.
    harness = BenignHarness(
        model_name=args.model,
        device=args.device,
        dtype_str=args.dtype,
        system_prompt=system_prompt,
        defense_prompt=defense_prompt,
        user_prompts=benign_prompts,
        gen_max_new=args.gen_max_new,
        inference_alpha=args.inference_alpha,
    )

    # 4) Generate outputs with the soft prompt, without it, and with the original prompt.
    soft_prompt = soft_prompt.to(harness.model_input_device)
    results = []
    csv_rows = []
    iterator = range(len(benign_prompts))
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(benign_prompts), unit="sample", desc="Benign test")

    for i in iterator:
        item = {
            "prompt": benign_prompts[i],
        }
        
        # Original output without the defense prompt or the soft prompt.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t00 = time.time()
        original_text = None
        if args.compare_original:
            original_text = harness.generate_original(i)
            item["original"] = original_text
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t11 = time.time()
        elapsed_original = t11 - t00
        print("Original Inference time is: ", elapsed_original)

        # Baseline output with the defense prompt and without the soft prompt.
        baseline_text = None
        if args.compare_baseline:
            baseline_text = harness.generate_baseline(i)
            item["baseline"] = baseline_text
        
        # Full output with both the defense prompt and the soft prompt.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with_soft = harness.generate_with_soft_prompt(soft_prompt, i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        print("Inference time is: ", elapsed)
        item["with_soft"] = with_soft
        results.append(item)

        # Record one CSV row.
        csv_rows.append({
            "model": args.model,
            "run_dir": args.run_dir,
            "inference_alpha": args.inference_alpha,
            "gen_max_new": args.gen_max_new,
            "system_prompt": system_prompt,
            "defense_prompt": defense_prompt,
            "user_prompt": benign_prompts[i],
            "original_output": original_text if original_text else "",
            "baseline_output": baseline_text if baseline_text else "",
            "with_soft_output": with_soft,
        })

    # 5) Print results to the console.
    print("\n=== Benign Test Results ===")
    for idx, r in enumerate(results, 1):
        print(f"\n---- Sample #{idx} ----")
        print(f"[User]\n{r['prompt']}\n")
        if args.compare_original and "original" in r:
            print(f"[Original (no defense, no soft)]\n{r['original'][:1000]}\n")
        if args.compare_baseline and "baseline" in r:
            print(f"[Baseline (with defense, no soft)]\n{r['baseline'][:1000]}\n")
        print(f"[With Soft (with defense, with soft)]\n{r['with_soft'][:1000]}\n")

    # 6) Save JSON results.
    out_path = args.output or os.path.join(args.run_dir, "test_outputs.json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "system_prompt": system_prompt,
            "defense_prompt": defense_prompt,
            "inference_alpha": args.inference_alpha,
            "gen_max_new": args.gen_max_new,
            "compare_baseline": bool(args.compare_baseline),
            "compare_original": bool(args.compare_original),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved JSON results to: {out_path}")

    # 7) Save CSV results.
    if args.csv_out is None:
        out_csv = os.path.join(args.run_dir, "test_outputs.csv")
    elif os.path.isabs(args.csv_out) or os.path.dirname(args.csv_out):
        out_csv = args.csv_out
    else:
        out_csv = os.path.join(args.run_dir, args.csv_out)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df = pd.DataFrame(csv_rows, columns=[
        "model", "run_dir", "inference_alpha", "gen_max_new",
        "system_prompt", "defense_prompt", "user_prompt",
        "original_output", "baseline_output", "with_soft_output"
    ])
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Saved CSV to: {out_csv}")


if __name__ == "__main__":
    main()