# test.py
'''
CUDA_VISIBLE_DEVICES=1 nohup python -u test_soft_prompt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --batch-run-root llama3_kl_1_atk_0.5_aligned \
  --sys-ids "6,7,8,9,10" \
  --benign-data-dir data/benign_data \
  --compare-original \
  --max-benign-prompts 20 \
  --benign-csv-out benign_result/llama3_kl_1_benign_eval.csv \
  --inference-alpha 1 \
  --disable-thinking \
  > benign_result/llama3_kl_1_benign_eval.log 2>&1 &
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


def apply_chat_template_compat(tokenizer, messages, disable_thinking: bool = False, **kwargs):
    if disable_thinking:
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def maybe_cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_sys_ids(raw: str) -> List[int]:
    ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    if not ids:
        raise ValueError("--sys-ids must contain at least one valid integer id")
    return ids


def load_prompt_list(json_path: str, max_prompts: int = 0) -> List[str]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, list) or not all(isinstance(x, str) for x in loaded):
        raise ValueError(f"{json_path} must be JSON list[str]")
    if max_prompts and max_prompts > 0:
        return loaded[:max_prompts]
    return loaded


def save_csv_rows(csv_rows: List[dict], out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df = pd.DataFrame(csv_rows, columns=[
        "model", "run_dir", "sys_id", "dataset_type",
        "inference_alpha", "gen_max_new",
        "system_prompt", "defense_prompt", "user_prompt",
        "original_output", "baseline_output", "with_soft_output"
    ])
    df.to_csv(out_csv, index=False, encoding="utf-8")


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
    Reuses the training-time concatenation and caching logic:
    System + Defense + {SOFT} + User.
    Supports baseline generation without the soft prompt, generation with the
    soft prompt inserted at the embedding level, and original generation without
    either the defense prompt or the soft prompt.
    """
    def __init__(
        self,
        model_name: Optional[str],
        device: str,
        dtype_str: str,
        system_prompt: str,
        defense_prompt: str,
        user_prompts: List[str],
        gen_max_new: int = 512,
        inference_alpha: float = 0.4,
        attention_type: str = "input_last_token",  # Kept as a compatibility placeholder.
        model=None,
        tokenizer=None,
        disable_thinking: bool = False,
    ):
        self.system_prompt = system_prompt
        self.defense_prompt = defense_prompt
        self.user_prompts = user_prompts
        self.gen_max_new = int(gen_max_new)
        self.inference_alpha = float(inference_alpha)
        self.disable_thinking = bool(disable_thinking)

        if model is None or tokenizer is None:
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
        else:
            self.model = model
            self.tokenizer = tokenizer

        self.device = device
        self.emb_layer = self.model.get_input_embeddings()
        self.model_input_device = self.emb_layer.weight.device

        self._prepare_static_parts()

    def _prepare_static_parts(self):
        tk = self.tokenizer
        soft_placeholder = "{SOFT}"

        # Build the template only to locate defense_slice.
        temp = apply_chat_template_compat(
            tk,
            [
                {"role": "system", "content": f"{self.system_prompt} {self.defense_prompt} {soft_placeholder}"},
                {"role": "user", "content": "PLACEHOLDER"},
            ],
            disable_thinking=self.disable_thinking,
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
            s = apply_chat_template_compat(
                tk,
                messages,
                disable_thinking=self.disable_thinking,
                tokenize=False,
                add_generation_prompt=True,
            )
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
        input_ids = apply_chat_template_compat(
            self.tokenizer,
            messages,
            disable_thinking=self.disable_thinking,
            add_generation_prompt=True,
            return_tensors="pt",
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
        input_ids = apply_chat_template_compat(
            self.tokenizer,
            messages,
            disable_thinking=self.disable_thinking,
            add_generation_prompt=True,
            return_tensors="pt",
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
    ap = argparse.ArgumentParser(description="Benign prompts sanity test with/without soft suffix.")
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
    ap.add_argument("--disable-thinking", action="store_true", help="Disable thinking mode when tokenizer supports it")
    ap.add_argument("--max-prompts", type=int, default=0, help="Limit prompt count in single-run mode")

    # Batch mode.
    ap.add_argument("--batch-run-root", type=str, default=None, help="Root folder containing id_x subfolders")
    ap.add_argument("--sys-ids", type=str, default="31,32,33,34,35,36,37,38,39,40")
    ap.add_argument("--attack-prompts-json", type=str, default=None)
    ap.add_argument("--benign-data-dir", type=str, default=None)
    ap.add_argument("--max-attack-prompts", type=int, default=100)
    ap.add_argument("--max-benign-prompts", type=int, default=40)
    ap.add_argument("--attack-csv-out", type=str, default=None)
    ap.add_argument("--benign-csv-out", type=str, default=None)

    # Output paths.
    ap.add_argument("--output", type=str, default=None, help="output json path; default: RUN_DIR/test_outputs.json")
    ap.add_argument("--csv-out", type=str, default=None, help="(CSV) output path; default: RUN_DIR/test_outputs.csv")
    return ap.parse_args()


def collect_single_run_prompts(args) -> List[str]:
    prompts: List[str] = []
    if args.prompts_json:
        prompts.extend(load_prompt_list(args.prompts_json, args.max_prompts))
    if args.prompt:
        for p in args.prompt:
            if isinstance(p, str) and p.strip():
                prompts.append(p.strip())
    if args.prompt_csv:
        df = pd.read_csv(args.prompt_csv)
        prompt_col = "test_user_prompt"
        if prompt_col not in df.columns:
            raise ValueError(
                f"--prompt-csv must contain a '{prompt_col}' column. "
                f"Available columns: {list(df.columns)}"
            )
        prompts.extend(
            df[prompt_col]
            .dropna()
            .astype(str)
            .map(str.strip)
            .loc[lambda values: values != ""]
            .tolist()
        )
        if args.max_prompts and args.max_prompts > 0:
            prompts = prompts[:args.max_prompts]
    if len(prompts) == 0:
        raise ValueError("No benign prompts provided. Use --prompts-json and/or --prompt.")
    return prompts


def run_evaluation(
    harness: BenignHarness,
    prompts: List[str],
    soft_prompt: Tensor,
    args,
    run_dir: str,
    system_prompt: str,
    defense_prompt: str,
    sys_id: Optional[str] = None,
    dataset_type: Optional[str] = None,
    progress_desc: str = "Benign test",
):
    soft_prompt = soft_prompt.to(harness.model_input_device)
    results = []
    csv_rows = []
    iterator = range(len(prompts))
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(prompts), unit="sample", desc=progress_desc)

    for i in iterator:
        item = {
            "prompt": prompts[i],
        }

        maybe_cuda_synchronize()
        t00 = time.time()
        original_text = None
        if args.compare_original:
            original_text = harness.generate_original(i)
            item["original"] = original_text
        maybe_cuda_synchronize()
        t11 = time.time()
        elapsed_original = t11 - t00
        print("Original Inference time is: ", elapsed_original)

        baseline_text = None
        if args.compare_baseline:
            baseline_text = harness.generate_baseline(i)
            item["baseline"] = baseline_text

        maybe_cuda_synchronize()
        t0 = time.time()
        with_soft = harness.generate_with_soft_prompt(soft_prompt, i)
        maybe_cuda_synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        print("Inference time is: ", elapsed)
        item["with_soft"] = with_soft
        results.append(item)

        csv_rows.append({
            "model": args.model,
            "run_dir": run_dir,
            "sys_id": sys_id or "",
            "dataset_type": dataset_type or "",
            "inference_alpha": args.inference_alpha,
            "gen_max_new": args.gen_max_new,
            "system_prompt": system_prompt,
            "defense_prompt": defense_prompt,
            "user_prompt": prompts[i],
            "original_output": original_text if original_text else "",
            "baseline_output": baseline_text if baseline_text else "",
            "with_soft_output": with_soft,
        })
    return results, csv_rows


def print_results(results, args):
    print("\n=== Benign Test Results ===")
    for idx, r in enumerate(results, 1):
        print(f"\n---- Sample #{idx} ----")
        print(f"[User]\n{r['prompt']}\n")
        if args.compare_original and "original" in r:
            print(f"[Original (no defense, no soft)]\n{r['original'][:1000]}\n")
        if args.compare_baseline and "baseline" in r:
            print(f"[Baseline (with defense, no soft)]\n{r['baseline'][:1000]}\n")
        print(f"[With Soft (with defense, with soft)]\n{r['with_soft'][:1000]}\n")


def run_single_mode(args):
    soft_prompt, system_prompt, defense_prompt = load_run_dir(
        run_dir=args.run_dir,
        fallback_system=args.system_prompt,
        fallback_defense=args.defense_prompt,
    )

    prompts = collect_single_run_prompts(args)
    harness = BenignHarness(
        model_name=args.model,
        device=args.device,
        dtype_str=args.dtype,
        system_prompt=system_prompt,
        defense_prompt=defense_prompt,
        user_prompts=prompts,
        gen_max_new=args.gen_max_new,
        inference_alpha=args.inference_alpha,
        disable_thinking=args.disable_thinking,
    )

    results, csv_rows = run_evaluation(
        harness=harness,
        prompts=prompts,
        soft_prompt=soft_prompt,
        args=args,
        run_dir=args.run_dir,
        system_prompt=system_prompt,
        defense_prompt=defense_prompt,
        progress_desc="Single test",
    )

    print_results(results, args)

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

    if args.csv_out is None:
        out_csv = os.path.join(args.run_dir, "test_outputs.csv")
    elif os.path.isabs(args.csv_out) or os.path.dirname(args.csv_out):
        out_csv = args.csv_out
    else:
        out_csv = os.path.join(args.run_dir, args.csv_out)
    save_csv_rows(csv_rows, out_csv)
    print(f"✅ Saved CSV to: {out_csv}")


def run_batch_mode(args):
    run_attack = bool(args.attack_prompts_json and args.attack_csv_out)
    run_benign = bool(args.benign_data_dir and args.benign_csv_out)
    if not run_attack and not run_benign:
        raise ValueError(
            "Batch mode requires attack args (--attack-prompts-json and --attack-csv-out), "
            "or benign args (--benign-data-dir and --benign-csv-out)."
        )

    selected_sys_ids = parse_sys_ids(args.sys_ids)
    attack_prompts = load_prompt_list(args.attack_prompts_json, args.max_attack_prompts) if run_attack else []

    dtype = getattr(torch, args.dtype)
    model_kwargs = {"torch_dtype": dtype}
    if str(args.device).startswith("cuda"):
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    all_attack_rows = []
    all_benign_rows = []

    for sys_id in selected_sys_ids:
        run_dir = os.path.join(args.batch_run_root, f"id_{sys_id}")
        summary_path = os.path.join(run_dir, "optimized_soft_prompt.json")
        soft_path = os.path.join(run_dir, "soft_prompt.pt")
        if not (os.path.exists(summary_path) and os.path.exists(soft_path)):
            print(f"⏭️  Skip id_{sys_id}: missing training outputs in {run_dir}")
            continue
        soft_prompt, system_prompt, defense_prompt = load_run_dir(
            run_dir=run_dir,
            fallback_system=args.system_prompt,
            fallback_defense=args.defense_prompt,
        )

        attack_harness = None
        benign_harness = None
        if run_attack:
            attack_harness = BenignHarness(
                model_name=None,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                dtype_str=args.dtype,
                system_prompt=system_prompt,
                defense_prompt=defense_prompt,
                user_prompts=attack_prompts,
                gen_max_new=args.gen_max_new,
                inference_alpha=args.inference_alpha,
                disable_thinking=args.disable_thinking,
            )
            _, attack_rows = run_evaluation(
                harness=attack_harness,
                prompts=attack_prompts,
                soft_prompt=soft_prompt,
                args=args,
                run_dir=run_dir,
                system_prompt=system_prompt,
                defense_prompt=defense_prompt,
                sys_id=str(sys_id),
                dataset_type="attack",
                progress_desc=f"Attack id_{sys_id}",
            )
            all_attack_rows.extend(attack_rows)

        if run_benign:
            benign_json = os.path.join(args.benign_data_dir, f"sys_id_{sys_id}.json")
            benign_prompts = load_prompt_list(benign_json, args.max_benign_prompts)
            benign_harness = BenignHarness(
                model_name=None,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                dtype_str=args.dtype,
                system_prompt=system_prompt,
                defense_prompt=defense_prompt,
                user_prompts=benign_prompts,
                gen_max_new=args.gen_max_new,
                inference_alpha=args.inference_alpha,
                disable_thinking=args.disable_thinking,
            )
            _, benign_rows = run_evaluation(
                harness=benign_harness,
                prompts=benign_prompts,
                soft_prompt=soft_prompt,
                args=args,
                run_dir=run_dir,
                system_prompt=system_prompt,
                defense_prompt=defense_prompt,
                sys_id=str(sys_id),
                dataset_type="benign",
                progress_desc=f"Benign id_{sys_id}",
            )
            all_benign_rows.extend(benign_rows)

        if attack_harness is not None:
            del attack_harness
        if benign_harness is not None:
            del benign_harness
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if run_attack:
        save_csv_rows(all_attack_rows, args.attack_csv_out)
        print(f"✅ Saved attack CSV to: {args.attack_csv_out}")
    if run_benign:
        save_csv_rows(all_benign_rows, args.benign_csv_out)
        print(f"✅ Saved benign CSV to: {args.benign_csv_out}")


def main():
    args = parse_args()
    if args.batch_run_root:
        run_batch_mode(args)
    else:
        run_single_mode(args)


if __name__ == "__main__":
    main()