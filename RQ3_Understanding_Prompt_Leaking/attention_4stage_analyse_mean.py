
import os, json, csv, math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.multiprocessing as mp
from multiprocessing import Queue, Manager
from queue import Empty
import threading
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
EPS = 1e-9


def prework(model_path="meta-llama/Llama-3.1-8B-Instruct",
            device="cuda", torch_dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype
    ).to(device).eval()
    return tok, mdl, device


def find_token_span_by_text(full_ids, target_text, tokenizer):
    full_text = tokenizer.decode(full_ids)
    st = full_text.find(target_text)
    if st == -1:
        return None
    before = full_text[:st]
    n_before = len(tokenizer(before, add_special_tokens=False).input_ids)
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
    return slice(n_before, n_before + len(target_ids))


class AttentionAnalyser:
    def __init__(self, tokenizer, model, device):
        self.tok, self.mdl, self.dev = tokenizer, model, device
        self.system_prompt = ""
        self.defense_suff = ""
        self.user_prompt = ""
        self.output = ""
        self.input_ids = None

        self._defense_slice = None
        self._user_slice = None
        self._assistant_role_slice = None
        self._target_slice = None

    def _encode_chat(self):
        msgs = [
            {"role": "system", "content": f"{self.system_prompt.strip()} {self.defense_suff.strip()}"},
            {"role": "user", "content": self.user_prompt.strip()},
        ]
        ids = self.tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)[0]
        self.input_ids = ids.cpu()

        sys_def = find_token_span_by_text(
            self.input_ids.tolist(),
            f"{self.system_prompt.strip()} {self.defense_suff.strip()}",
            self.tok
        )
        sys_only = find_token_span_by_text(
            self.input_ids.tolist(),
            self.system_prompt.strip(),
            self.tok
        )
        if sys_def is None or sys_only is None:
            raise ValueError("Cannot locate system/defense slice.")
        self._defense_slice = slice(sys_only.stop, sys_def.stop)

        self._user_slice = find_token_span_by_text(
            self.input_ids.tolist(),
            self.user_prompt.strip(),
            self.tok
        )
        if self._user_slice is None:
            raise ValueError("Cannot locate user slice.")

        self._assistant_role_slice = slice(self._user_slice.stop, len(self.input_ids))
        self._target_slice = slice(self._assistant_role_slice.stop, len(self.input_ids))

    def _generate_once(self, max_new_tokens=256, temperature=None):
        gen_cfg = self.mdl.generation_config
        gen_cfg.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_cfg.temperature = float(temperature)

        ids = self.input_ids[:self._assistant_role_slice.stop].to(self.dev).unsqueeze(0)
        mask = torch.ones_like(ids, device=self.dev)
        with torch.no_grad():
            out = self.mdl.generate(
                ids,
                attention_mask=mask,
                generation_config=gen_cfg,
                pad_token_id=self.tok.pad_token_id
            )[0]
        gen_ids = out[self._assistant_role_slice.stop:]
        self.output = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

        msgs = [
            {"role": "system", "content": f"{self.system_prompt.strip()} {self.defense_suff.strip()}"},
            {"role": "user", "content": self.user_prompt.strip()},
            {"role": "assistant", "content": self.output}
        ]
        ids_full = self.tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=False)[0]
        self.input_ids = ids_full.cpu()

        self._target_slice = slice(self._assistant_role_slice.stop, len(self.input_ids))

    def _mean_scores_for_segments(self, scores_1d, stage_name="stage"):

        atk_vec = scores_1d[self._user_slice]
        def_vec = scores_1d[self._defense_slice]

        res = {}
        

        if len(atk_vec) > 0:
            res[f"{stage_name}_atk_mean"] = float(atk_vec.mean().cpu())
        else:
            res[f"{stage_name}_atk_mean"] = np.nan
        

        if len(def_vec) > 0:
            res[f"{stage_name}_def_mean"] = float(def_vec.mean().cpu())
        else:
            res[f"{stage_name}_def_mean"] = np.nan
        
        if len(atk_vec) > 0 and len(def_vec) > 0:
            res[f"{stage_name}_atk_minus_def"] = res[f"{stage_name}_atk_mean"] - res[f"{stage_name}_def_mean"]
        else:
            res[f"{stage_name}_atk_minus_def"] = np.nan

        return res
    def compute_logits_bias_stats(self):
        
        with torch.no_grad():
            outs = self.mdl(
                input_ids=self.input_ids.to(self.dev).unsqueeze(0),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_in_last = outs.hidden_states[-2][0]   # (seq_len, hidden_dim)
        last_layer = self.mdl.model.layers[-1]
        attn_mod = last_layer.self_attn


        hidden_ln_last = last_layer.input_layernorm(hidden_in_last)  # (seq_len, hidden_dim)
        h_batch_ln = hidden_ln_last.unsqueeze(0)                     # (1, seq_len, hidden_dim)
        bsz, q_len, _ = h_batch_ln.size()


        q_states = attn_mod.q_proj(h_batch_ln)  # (1, seq_len, hidden_dim)
        k_states = attn_mod.k_proj(h_batch_ln)  # (1, seq_len, hidden_dim)


        q_states = q_states.view(
            bsz, q_len, attn_mod.num_heads, attn_mod.head_dim
        ).transpose(1, 2)  # (1, n_heads, seq_len, head_dim)

        k_states = k_states.view(
            bsz, q_len, attn_mod.num_key_value_heads, attn_mod.head_dim
        ).transpose(1, 2)  # (1, n_kv_heads, seq_len, head_dim)

        # GQA
        if attn_mod.num_key_value_heads != attn_mod.num_heads:
            k_states = _repeat_kv(
                k_states, attn_mod.num_heads // attn_mod.num_key_value_heads
            )  # (1, n_heads, seq_len, head_dim)

        # RoPE
        position_ids = torch.arange(
            0, q_len, dtype=torch.long, device=h_batch_ln.device
        ).unsqueeze(0)  # (1, seq_len)

        cos, sin = attn_mod.rotary_emb(k_states, position_ids)
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        # logits matrix
        logits_all = torch.matmul(
            q_states, k_states.transpose(-2, -1)
        ) / math.sqrt(attn_mod.head_dim)


        logits_all = logits_all.mean(dim=1)[0]


        t0 = self._target_slice.start
        logits_all = logits_all[:t0, :t0]               # (T_in, T_in)


        mean_over_q = logits_all.mean(dim=0)            # (T_in,)
        pos_frac_over_q = (logits_all > 0).float().mean(dim=0)  # (T_in,)

        pos_mask = (logits_all > 0)
        global_pos_count = int(pos_mask.sum().item())
        global_total_count = int(logits_all.numel())
        global_pos_frac = float(global_pos_count / (global_total_count + EPS))

        stats = {
            "logits_global_mean": float(logits_all.mean().cpu()),
            "logits_global_min": float(logits_all.min().cpu()),
            "logits_global_max": float(logits_all.max().cpu()),
            "logits_global_pos_count": global_pos_count,
            "logits_global_total_count": global_total_count,
            "logits_global_pos_frac": global_pos_frac,
        }
        
        def _segment_stats(slice_obj, name_prefix):
            cols_mean = mean_over_q[slice_obj]         
            cols_frac = pos_frac_over_q[slice_obj]

            if len(cols_mean) > 0:
                stats[f"{name_prefix}_mean_logits_col"] = float(cols_mean.mean().cpu())
                stats[f"{name_prefix}_pos_frac_col"] = float(cols_frac.mean().cpu())
            else:
                stats[f"{name_prefix}_mean_logits_col"] = np.nan
                stats[f"{name_prefix}_pos_frac_col"] = np.nan

            sub_logits = logits_all[:, slice_obj]        # (T_in, #cols_in_segment)
            sub_mask = sub_logits > 0

            seg_pos_count = int(sub_mask.sum().item())
            seg_total_count = int(sub_mask.numel())
            seg_pos_frac = float(seg_pos_count / (seg_total_count + EPS)) if seg_total_count > 0 else np.nan

            if seg_pos_count > 0:
                seg_pos_mean = float(sub_logits[sub_mask].mean().cpu())
            else:
                seg_pos_mean = np.nan

            stats[f"{name_prefix}_pos_count_all"] = seg_pos_count
            stats[f"{name_prefix}_total_count_all"] = seg_total_count
            stats[f"{name_prefix}_pos_frac_all"] = seg_pos_frac
            stats[f"{name_prefix}_pos_mean_all"] = seg_pos_mean  
        _segment_stats(self._user_slice, "user")
        _segment_stats(self._defense_slice, "defense")
        return stats, logits_all.cpu(), mean_over_q.cpu(), pos_frac_over_q.cpu()


    def compute_4stage_means(self):
        """
        Compute four stages:
        - Stage 1: Original hidden states X (input to the last layer, before layer normalization).
        - Stage 2: L2 norm of K (computed from the K vectors after the last layer input_layernorm,
                without applying RoPE, averaged over KV heads).
        - Stage 3: True attention logits ≈ q·k / √d as used in the model
                (computed from Q/K after the last layer input_layernorm,
                with GQA and RoPE applied, averaged over attention heads).
        - Stage 4: Attention weights (the last-layer attention output from the model forward pass,
                after softmax, averaged over attention heads).

        For each stage, the mean values are computed separately for the attack segment
        and the defense segment.
        """
        with torch.no_grad():
            outs = self.mdl(
                input_ids=self.input_ids.to(self.dev).unsqueeze(0),
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )

        # --- Extract information from the last layer ---
        # The structure of hidden_states is:
        # [embedding_out, layer0_out, ..., layer_{L-1}_out]
        # Therefore, the second-to-last element corresponds to the input of the last layer,
        # while the last element corresponds to the output of the last layer.

        hidden_in_last = outs.hidden_states[-2][0]   # (seq_len, hidden_dim)
        last_attn = outs.attentions[-1][0]           # (num_heads, tgt_len, src_len)
        attn_mean = last_attn.mean(0)                # (tgt_len, src_len)

        # Target token: The position of the first generated token in the sequence.
        t0 = self._target_slice.start

        # ===== Stage 1: X Hidden State =====
        norms_x = torch.norm(hidden_in_last, p=2, dim=-1)  # (seq_len,)
        norms_x = norms_x[:t0]
        res_stage1 = self._mean_scores_for_segments(norms_x, stage_name="stage1")

        last_layer = self.mdl.model.layers[-1]
        attn_mod = last_layer.self_attn

        # First perform input_layernorm, then perform q_proj/k_proj.
        hidden_ln_last = last_layer.input_layernorm(hidden_in_last)  # (seq_len, hidden_dim)
        h_batch_ln = hidden_ln_last.unsqueeze(0)                     # (1, seq_len, hidden_dim)
        bsz, q_len, _ = h_batch_ln.size()

        # ===== Stage 2: Key-Vector Projection =====
        k_states2 = attn_mod.k_proj(h_batch_ln)   # (1, seq_len, hidden_dim)
        k_states2 = k_states2.view(
            bsz, q_len, attn_mod.num_key_value_heads, attn_mod.head_dim
        ).transpose(1, 2)                         # (1, n_kv_heads, seq_len, head_dim)

        k_mean2 = k_states2.mean(dim=1)[0]        # (seq_len, head_dim)
        norms_k = torch.norm(k_mean2, p=2, dim=-1)  # (seq_len,)
        norms_k = norms_k[:t0]
        res_stage2 = self._mean_scores_for_segments(norms_k, stage_name="stage2")

        # ===== Stage 3: Q/K → RoPE → logits =====
        # 1) Q / K
        q_states = attn_mod.q_proj(h_batch_ln)  # (1, seq_len, hidden_dim)
        k_states = attn_mod.k_proj(h_batch_ln)  # (1, seq_len, hidden_dim)

        # 2) multi head + GQA
        q_states = q_states.view(
            bsz, q_len, attn_mod.num_heads, attn_mod.head_dim
        ).transpose(1, 2)  # (1, n_heads, seq_len, head_dim)

        k_states = k_states.view(
            bsz, q_len, attn_mod.num_key_value_heads, attn_mod.head_dim
        ).transpose(1, 2)  # (1, n_kv_heads, seq_len, head_dim)


        if attn_mod.num_key_value_heads != attn_mod.num_heads:
            k_states = _repeat_kv(
                k_states, attn_mod.num_heads // attn_mod.num_key_value_heads
            )  # (1, n_heads, seq_len, head_dim)

        # 3) RoPE
        position_ids = torch.arange(
            0, q_len, dtype=torch.long, device=h_batch_ln.device
        ).unsqueeze(0)  # (1, seq_len)

        cos, sin = attn_mod.rotary_emb(k_states, position_ids)
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        # 4) Take only the query (t0) that generates the first token, and perform a dot product with the K of all historical tokens.
        # q_t0: (1, n_heads, 1, head_dim)
        q_t0 = q_states[:, :, t0:t0+1, :]
        # logits_per_head: (1, n_heads, 1, seq_len)
        logits_per_head = torch.matmul(
            q_t0, k_states.transpose(-2, -1)
        ) / math.sqrt(attn_mod.head_dim)

        logits_per_head = logits_per_head.squeeze(0).squeeze(1)  # (n_heads, seq_len)
        logits_mean = logits_per_head.mean(dim=0)                 # (seq_len,)
        logits_mean = logits_mean[:t0]                            

        res_stage3 = self._mean_scores_for_segments(logits_mean, stage_name="stage3")

        atk_vec = logits_mean[self._user_slice]
        def_vec = logits_mean[self._defense_slice]

        atk_pos = atk_vec[atk_vec > 0]
        atk_neg = atk_vec[atk_vec < 0]
        res_stage3["stage3_atk_pos_mean"] = (
            float(atk_pos.mean().cpu()) if len(atk_pos) > 0 else np.nan
        )
        res_stage3["stage3_atk_pos_count"] = int(len(atk_pos))
        res_stage3["stage3_atk_neg_count"] = int(len(atk_neg))

        def_pos = def_vec[def_vec > 0]
        def_neg = def_vec[def_vec < 0]
        res_stage3["stage3_def_pos_mean"] = (
            float(def_pos.mean().cpu()) if len(def_pos) > 0 else np.nan
        )
        res_stage3["stage3_def_pos_count"] = int(len(def_pos))
        res_stage3["stage3_def_neg_count"] = int(len(def_neg))

        # ===== Stage 4: attention weights =====
        attn_t0 = attn_mean[t0, :t0]  # (seq_len_input,)
        res_stage4 = self._mean_scores_for_segments(attn_t0, stage_name="stage4")

        all_res = {}
        all_res.update(res_stage1)
        all_res.update(res_stage2)
        all_res.update(res_stage3)
        all_res.update(res_stage4)
        return all_res

    

def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
    return hidden_states.reshape(hidden_states.size(0), hidden_states.size(1) * n_rep, hidden_states.size(3), hidden_states.size(4))

def run_once(tok, mdl, dev, system_prompt, defensive_instruction, user_prompt):
    ana = AttentionAnalyser(tok, mdl, dev)
    ana.system_prompt = system_prompt
    ana.defense_suff = defensive_instruction
    ana.user_prompt = user_prompt

    ana._encode_chat()
    ana._generate_once(max_new_tokens=256)

    means_4stage = ana.compute_4stage_means()
    logits_bias_stats, _, _, _ = ana.compute_logits_bias_stats()
    return {
        "system_prompt": system_prompt,
        "defensive_instruction": defensive_instruction,
        "user_prompt": user_prompt,
        "output": ana.output,
        **means_4stage,
        **logits_bias_stats,  
    }

def worker_process(gpu_id, task_queue, result_queue, model_path):

    device = f"cuda:{gpu_id}"
    print(f"[Worker-GPU{gpu_id}] init...")
    
    try:
        tok, mdl, dev = prework(model_path=model_path, device=device)
        print(f"[Worker-GPU{gpu_id}] Model loading complete, task processing begins.")
        
        while True:
            try:

                task = task_queue.get(timeout=1)
                
                if task is None:
                    print(f"[Worker-GPU{gpu_id}] Received end signal")
                    break
                
                si, di, ui, sp, df_, up, model_name = task
                
                try:
                    row = run_once(tok, mdl, dev, sp, df_, up)
                    row = {
                        "model": model_name, 
                        "sys_idx": si, 
                        "def_idx": di, 
                        "user_idx": ui, 
                        **row
                    }
                    result_queue.put(("success", row))
                except Exception as e:
                    result_queue.put(("error", (si, di, ui, str(e))))
                    
            except Empty:
                continue
                
    except Exception as e:
        print(f"[Worker-GPU{gpu_id}] initialization failed {e}")
    finally:
        print(f"[Worker-GPU{gpu_id}] exit")


def writer_thread(result_queue, out_csv, header, total_tasks):

    write_header = not os.path.exists(out_csv)
    
    with open(out_csv, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=header)
        if write_header:
            writer.writeheader()
        
        completed = 0
        errors = 0
        
        while completed + errors < total_tasks:
            try:
                result_type, data = result_queue.get(timeout=1)
                
                if result_type == "success":
                    writer.writerow(data)
                    completed += 1
                elif result_type == "error":
                    si, di, ui, error_msg = data
                    print(f"[WARN] task failed ({si},{di},{ui}): {error_msg}")
                    errors += 1
                
                if (completed + errors) % 50 == 0:
                    print(f"[progress] {completed + errors}/{total_tasks} (sucess={completed}, fail={errors})")
                    
            except Empty:
                continue
    
    print(f"Finish! Total: sucess={completed}, fail={errors}")


if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)
    

    GPU_DEVICES = [0]
    WORKERS_PER_GPU = 1
    

    system_prompts_csv = "data/system_prompts.csv"
    user_prompts_json  = "data/llama3_adversarial_query.json"
    defense_suffs_json = "data/defense.json"
    out_dir = "ft_mean_outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "4stage_mean_parallel.csv")
    

    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "Llama-3.1-8B-Instruct"
    

    sys_df = pd.read_csv(system_prompts_csv)

    system_prompts = sys_df["system_prompt"].dropna().tolist()
    
    with open(user_prompts_json, "r", encoding="utf-8") as f:
        up_raw = json.load(f)
    user_prompts = (list(up_raw.values()) if isinstance(up_raw, dict)
                    else [x["text"] if isinstance(x, dict) and "text" in x else str(x) for x in up_raw])
    
    with open(defense_suffs_json, "r", encoding="utf-8") as f:
        dp_raw = json.load(f)
    defense_suffs = (list(dp_raw.values()) if isinstance(dp_raw, dict)
                     else [x["text"] if isinstance(x, dict) and "text" in x else str(x) for x in dp_raw])
    
    stage_metrics = []
    for stage in ["stage1", "stage2", "stage3", "stage4"]:
        stage_metrics.extend([
            f"{stage}_atk_mean",
            f"{stage}_def_mean",
            f"{stage}_atk_minus_def",
        ])
    
    header = [
        "model","sys_idx","def_idx","user_idx",
        "system_prompt","defensive_instruction","user_prompt","output",

        "stage3_atk_pos_mean", "stage3_atk_pos_count","stage3_atk_neg_count",
        "stage3_def_pos_mean", "stage3_def_pos_count","stage3_def_neg_count",

        "logits_global_mean","logits_global_min","logits_global_max",
        "logits_global_pos_count","logits_global_total_count","logits_global_pos_frac",
        "user_mean_logits_col","user_pos_frac_col",
        "user_pos_count_all","user_total_count_all","user_pos_frac_all","user_pos_mean_all", 
        "defense_mean_logits_col","defense_pos_frac_col",
        "defense_pos_count_all","defense_total_count_all","defense_pos_frac_all","defense_pos_mean_all",
        *stage_metrics
    ]


    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    total_tasks = 0
    for si, sp in enumerate(system_prompts):
        for di, df_ in enumerate(defense_suffs):
            for ui, up in enumerate(user_prompts):
                task_queue.put((si, di, ui, sp, df_, up, model_name))
                total_tasks += 1
    
    print(f"[INFO] Total tasks: {total_tasks}")
    print(f"[INFO] GPU config: {len(GPU_DEVICES)} GPU，each GPU {WORKERS_PER_GPU} worker")
    print(f"[INFO] Total worker: {len(GPU_DEVICES) * WORKERS_PER_GPU}")
    

    writer = threading.Thread(
        target=writer_thread,
        args=(result_queue, out_csv, header, total_tasks)
    )
    writer.start()
    

    processes = []
    for gpu_id in GPU_DEVICES:
        for _ in range(WORKERS_PER_GPU):
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, task_queue, result_queue, model_path)
            )
            p.start()
            processes.append(p)
    

    for _ in processes:
        task_queue.put(None)
    

    print("[INFO] wait worker finish...")
    for p in processes:
        p.join()
    
    writer.join()
    
    print(f"[DONE] result are saved to {out_csv}")