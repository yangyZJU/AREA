import os
import json
import pandas as pd
import numpy as np
import torch
import math
import csv
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM)

word_show = 50


def prework():
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    device = 'cuda'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device).eval()

    return tokenizer, model



def aggregate_segment_attention(attn_vec, slice_obj, mode="mean", k=5):
    segment = attn_vec[slice_obj]
    if mode == "mean":
        return segment.mean().item()
    elif mode == "max":
        return segment.max().item()
    elif mode == "topk":
        k = min(k, segment.shape[0])
        top_vals, _ = torch.topk(segment, k)
        return top_vals.mean().item()
    else:
        raise ValueError(f"Unsupported mode: {mode}")



def find_token_span_by_text(full_ids, target_text, tokenizer):
    full_text = tokenizer.decode(full_ids)
    start_char = full_text.find(target_text)
    if start_char == -1:
        return None

    before = full_text[:start_char]
    num_tokens_before = len(tokenizer(before, add_special_tokens=False).input_ids)
    target_token_ids = tokenizer(target_text, add_special_tokens=False).input_ids
    return slice(num_tokens_before-1, num_tokens_before-1 + len(target_token_ids))


class AttentionAnalyse(object):
    def __init__(self,
        tokenizer,
        model
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.system_prompt = ''
        self.defense_suff = ''
        self.user_prompt = ''
        self.goal = ''
        self.output = ''

    def generate(self):
        gen_config = self.model.generation_config
        gen_config.max_new_tokens = 256
        #gen_config.temperature = 0.1
        input_ids = self.input_ids[:self._assistant_role_slice.stop].to(self.model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attn_masks,
            generation_config=gen_config,
            pad_token_id=self.tokenizer.pad_token_id
        )[0]
        self.output_ids = output_ids[self._assistant_role_slice.stop:]
        self.output = self.tokenizer.decode(self.output_ids, skip_special_tokens=True)
        # print('output_ids: ',self.output_ids)
        print('output: ', self.output)
    def update_ids(self, use_generate=True):
        target = self.output if use_generate else self.goal

        messages = [
            {"role": "system", "content": f"{self.system_prompt.strip()} {self.defense_suff.strip()}"},
            {"role": "user", "content": self.user_prompt.strip()},
        ]

        has_output = use_generate and self.output.strip()
        

        
        add_gen_prompt = not has_output

        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )[0]
        self.input_ids = input_ids.cpu()

        self._system_defense_slice = find_token_span_by_text(
            self.input_ids.tolist(),
            f"{self.system_prompt.strip()} {self.defense_suff.strip()}",
            self.tokenizer
        )
        self._system_prompt_slice = find_token_span_by_text(
            self.input_ids.tolist(),
            self.system_prompt.strip(),
            self.tokenizer
        )
        self._defense_slice = slice(self._system_prompt_slice.stop, self._system_defense_slice.stop)
        self._user_slice = find_token_span_by_text(
            self.input_ids.tolist(),
            self.user_prompt.strip(),
            self.tokenizer
        )

        self._assistant_role_slice = slice(self._user_slice.stop, len(self.input_ids))
        self._target_slice = slice(self._assistant_role_slice.stop, len(self.input_ids))

        if has_output:
            messages.append({"role": "assistant", "content": self.output.strip()})
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=False
            )[0]
            self.input_ids = input_ids.cpu()
            self._target_slice = slice(self._assistant_role_slice.stop, len(self.input_ids))


    def debug_input_slices(self):
        print("======== Slice Debugging ========")
        input_text = self.tokenizer.decode(self.input_ids)
        print(f"[Full Decoded Input] ({len(self.input_ids)} tokens):")
        print(input_text)
        print("----------------------------------")

        def show_slice(name, sl):
            token_ids = self.input_ids[sl]
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            text = self.tokenizer.decode(token_ids)
            print(f"[{name}] Slice {sl.start}:{sl.stop} ({len(token_ids)} tokens)")
            print(f"Tokens: {tokens}")
            print(f"Text  : {text}")
            print("")

        show_slice("System Prompt", self._system_prompt_slice)
        show_slice("Defensive Instruction", self._defense_slice)
        show_slice("User Query", self._user_slice)
        show_slice("Assistant Role Token", self._assistant_role_slice)
        show_slice("Target Output", self._target_slice)


    def get_segment_token_attention_sink(self, top_k=5, plot=True, save_path="segment_token_sink.png"):
        
        input_ids = self.input_ids.to(self.model.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_attentions=True)
            attentions = outputs.attentions

        
        last_layer_attn = attentions[-1][0].mean(0)  # shape: (tgt_len, src_len)

        def_idx = list(range(self._defense_slice.start, self._defense_slice.stop))
        user_idx = list(range(self._user_slice.start, self._user_slice.stop))
        combined_idx = def_idx + user_idx

        attn_to_combined = last_layer_attn[self._target_slice][:, combined_idx]  # (tgt_len, combined_len)
        token_total_attn = attn_to_combined.sum(dim=0)
        token_total_attn = F.normalize(token_total_attn, p=1, dim=0)

        token_ids = self.input_ids[combined_idx]
        token_texts = self.tokenizer.convert_ids_to_tokens(token_ids)
        token_texts = [t.replace("Ġ", " ").replace("▁", " ").strip() for t in token_texts]
        segment_flags = ['D'] * len(def_idx) + ['U'] * len(user_idx)

        top_vals, top_idxs = torch.topk(token_total_attn, top_k)

        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import numpy as np

            colors = ['blue' if f == 'D' else 'red' for f in segment_flags]
            x = range(len(token_texts))
            y = token_total_attn.cpu().numpy()

            plt.figure(figsize=(14, 4))
            plt.bar(x, y, color=colors)
            plt.xticks(x, token_texts, rotation=60)
            plt.xlabel("Prompt Tokens (D:Defense, U:User)")
            plt.ylabel("Total Attention from Output Tokens")
            #plt.title("Token Attention Distribution in Defense + User Query")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return {
            "tokens": token_texts,
            "attn_scores": token_total_attn.cpu().tolist(),
            "segments": segment_flags,
            "top_indices": top_idxs.cpu().tolist(),
            "top_tokens": [token_texts[i] for i in top_idxs],
            "top_scores": top_vals.cpu().tolist()
        }

    def attention_layerwise_tokens(self, top_k_tokens=2, save_dir="./attn", attention_type = "mean", name_prefix=""):
        input_ids = self.input_ids.to(self.model.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_attentions=True)
            attentions = outputs.attentions  # List of (batch, head, tgt_len, src_len)

        num_layers = len(attentions)
        target_token_ids = self.output_ids[:top_k_tokens]
        results = []
        FT_DAR_list = [] 
        for token_idx, output_token_id in enumerate(target_token_ids):
            attn_across_layers = []
            for layer_idx in range(num_layers - 1):
                # Shape: (head, tgt_len, src_len)
                layer_attn = attentions[layer_idx][0]
                # Extract attention for current token across all heads → (head, src_len)
                token_attn = layer_attn[:, token_idx + self._target_slice.start, :]
                avg_attn = token_attn.mean(0)  # (src_len,)
                norm_attn = F.normalize(avg_attn, p=1, dim=0)

                sp_score = aggregate_segment_attention(norm_attn, self._system_prompt_slice, mode=attention_type)
                #sp_score = aggregate_segment_attention(norm_attn, self._system_prompt_slice, mode="mean")#
                dp_score = aggregate_segment_attention(norm_attn, self._defense_slice, mode=attention_type)
                up_score = aggregate_segment_attention(norm_attn, self._user_slice, mode=attention_type)
                

                attn_across_layers.append([sp_score, dp_score, up_score])

            token_attn_matrix = np.array(attn_across_layers).T

            dp_mean = token_attn_matrix[1].mean()
            up_mean = token_attn_matrix[2].mean()
            FT_DAR_mean = dp_mean/(up_mean+dp_mean)
            FT_DAR_list.append(FT_DAR_mean)


            results.append((token_idx, output_token_id.item(), token_attn_matrix))
            
        return FT_DAR_list
    def attention_calc(self, attention_type, k=5):
        input_ids = self.input_ids.to(self.model.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_attentions=True)
            attentions = outputs.attentions
            
        last_layer_attn = attentions[-1][0]
       
        attn = last_layer_attn.mean(0)[self._target_slice]
        
        attn = F.normalize(attn, p=1,dim=1)

        if attention_type == "mean":
            attn_system = attn[:, self._system_prompt_slice].mean(1)
            attn_defense = attn[:, self._defense_slice].mean(1)
            attn_user = attn[:, self._user_slice].mean(1)
        elif attention_type == "max":
            attn_system = attn[:, self._system_prompt_slice].max(1).values
            attn_defense = attn[:, self._defense_slice].max(1).values
            attn_user = attn[:, self._user_slice].max(1).values
        elif attention_type == "topk":
            sorted_vals, _ = attn[:, self._system_prompt_slice].topk(k, dim=1)
            attn_system = sorted_vals.mean(1)

            sorted_vals, _ = attn[:, self._defense_slice].topk(k, dim=1)
            attn_defense = sorted_vals.mean(1)

            sorted_vals, _ = attn[:, self._user_slice].topk(k, dim=1)
            attn_user = sorted_vals.mean(1)

        data = [attn_system.cpu().tolist(), attn_defense.cpu().tolist(), attn_user.cpu().tolist()]
        data = [item[:word_show] if len(item) >= word_show else item for item in data]
        x_ids_labels = self.output_ids[:word_show] if len(self.output_ids) >= word_show else self.output_ids
        x_labels = [self.tokenizer.decode(item) for item in x_ids_labels]
        print(x_labels)
        y_labels = ['System Prompt', 'Defensive Instruction', 'User Query']

        return data, x_labels, y_labels
    def save_full_prompt_output(self, save_path="output_dump.txt"):


        lines = [
            "====== System Prompt ======",
            self.system_prompt.strip(),
            "",
            "====== Defensive Instruction ======",
            self.defense_suff.strip(),
            "",
            "====== User Query ======",
            self.user_prompt.strip(),
            "",
            "====== Model Output ======",
            self.output.strip(),
            ""
        ]

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[✔] Prompt + Output saved to: {save_path}")

    def getAttention(self,
        system_prompt, 
        defense_suff, 
        user_prompt,
        attention_type,
        use_generate = True,
        goal = ''
    ):
        self.system_prompt = system_prompt
        self.defense_suff = defense_suff
        self.user_prompt = user_prompt
        self.goal = goal
        self.output = ''
        if use_generate:
            self.update_ids(use_generate)
            self.generate()
        self.update_ids(use_generate)

        self.debug_input_slices()
        return self.attention_calc(attention_type)

    def compute_DAR_statistics(self, save_csv="DAR_stats_llama3.csv", k_list=[3,5,7,9]):

        save_dir = os.path.dirname(save_csv)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        epsilon = 1e-12

        input_ids = self.input_ids.to(self.model.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_attentions=True)
            attn = outputs.attentions[-1][0].mean(0)[self._target_slice]  # [T_out, T_all]
            attn = F.normalize(attn, p=1, dim=1)
        attn_def = attn[:, self._defense_slice].mean(1) 
        attn_usr = attn[:, self._user_slice].mean(1) 

        #cal mean DAR
        mean_def = attn_def.mean().item()
        mean_usr = attn_usr.mean().item()
        DAR_mean = mean_def / (mean_usr + mean_def + epsilon)

        #cal top-k DAR
        DAR_topk = {}
        for k in k_list:
            topk_def = attn_def.topk(min(k, len(attn_def)))[0].mean().item()
            topk_usr = attn_usr.topk(min(k, len(attn_usr)))[0].mean().item()
            DAR_topk[f"DARtop{k}"] = topk_def / (topk_usr + topk_def + epsilon)

        FT_DAR_list = self.attention_layerwise_tokens(top_k_tokens=2)

        FT_DAR_1 = FT_DAR_list[0] if len(FT_DAR_list) > 0 else None
        FT_DAR_2 = FT_DAR_list[1] if len(FT_DAR_list) > 1 else None
        result = {
            "system_prompt": self.system_prompt.strip(),
            "defense_suffix": self.defense_suff.strip(),
            "user_prompt": self.user_prompt.strip(),
            "output": self.output.strip(),
            "DARmean": DAR_mean,
            "FT_DAR_mean1": FT_DAR_1,
            "FT_DAR_mean2": FT_DAR_2
        }
        result.update(DAR_topk)

        write_header = not os.path.exists(save_csv)
        with open(save_csv, "a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(result)

        print(f"DAR statistics saved to {save_csv}")
        return result
if __name__ == "__main__":
    system_prompts_csv = 'data/system_prompts.csv' 
    user_prompts_json = 'data/llama3_adversarial_query.json'
    defensive_instruction_json = 'data/defense.json'
    output_dir = 'attn_llama3'    

    os.makedirs(output_dir, exist_ok=True)

    system_prompts_df = pd.read_csv(system_prompts_csv)
    system_prompts = system_prompts_df['system_prompt'].dropna().tolist()

    with open(user_prompts_json, 'r', encoding='utf-8') as f:
        user_prompts = json.load(f)
    user_prompts = user_prompts
    with open(defensive_instruction_json, 'r', encoding='utf-8') as f:
        defensive_instruction = json.load(f)
    defensive_instruction = defensive_instruction
    
    results1 = []
    results2 = []
    tokenizer, model = prework()

    attentionAnalyse = AttentionAnalyse(tokenizer, model)
    attention_type = "mean"


    for sys_index, system_prompt in enumerate(system_prompts):
        for def_index, defense_suff in enumerate(defensive_instruction):
            for user_index, user_prompt in enumerate(user_prompts):
                try:
                    data, x_labels, y_labels = attentionAnalyse.getAttention(system_prompt, defense_suff, user_prompt, attention_type, use_generate = True)
                except Exception as e:
                    print(f"[WARN] failed on sys={sys_index}, def={def_index}, user={user_index}: {e}")
                    continue
                if def_index % 10 == 0:
                    plt.figure(figsize=(18, 3))
                    sns.heatmap(data, cmap='coolwarm', annot=False, fmt=".2f", xticklabels=x_labels, yticklabels=y_labels)

                    plt.xticks(rotation=60, fontsize=15, ha='right')
                    plt.yticks(fontsize=15)

                    plt.tight_layout()
                    plt.savefig(f'./attn_llama3/heatMap_{sys_index}_{def_index}_{user_index}.png')

                DAR_res = attentionAnalyse.compute_DAR_statistics(
                    save_csv="./results/llama3_DAR_results.csv", 
                    k_list=[3,5,7,9]
                )
                print(DAR_res)