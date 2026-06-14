import json
import logging
import sys
import torch

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from datasets import Dataset
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainingArguments

from data.loader import load_and_prepare_dataset
from data.utils import TextDataset, create_collate_fn
from src.finetuning_utils import GpuMemoryCallbackIntegrated, ManualAdapterSaveCallback, CustomDataCollatorForLanguageModeling
from src.logging_config import setup_logging
from src.model import Model
from src.output_generation import precompute_model_outputs
from src.style_prompts import get_style_prompt
from src.utils import set_seed





def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Script for finetuning.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Huggingface model name to use for finetuning"
    )
    quantization_group = parser.add_mutually_exclusive_group(required=False)
    quantization_group.add_argument(
        "--quantize_4bit",
        action="store_true",
        help="Enable 4-bit quantization for the model. (Cannot be used with --quantize_8bit)"
    )
    quantization_group.add_argument(
        "--quantize_8bit",
        action="store_true",
        help="Enable 8-bit quantization for the model. (Cannot be used with --quantize_4bit)"
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Specify a custom system prompt directly as a string."
    )
    prompt_group.add_argument(
        "--style",
        type=str,
        default=None,
        help=(
            "Specify a predefined style for the system prompt.\n"
            "The available styles are defined in 'style_prompts.py'."
        )
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=100,
        help="Dataset size for optimization (80:20 split)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="truthfulqa",
        choices = ["truthfulqa", "triviaqa", "cnn_dailymail", "samsum"],
        help="Dataset to use for optimization"    
    )
    parser.add_argument(
        "--task_hints",
        default=False,
        action="store_true",
        help="Whether to use task hints"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for optimization"
    )
    parser.add_argument(
        "--output_token_count",
        type=int,
        default=15,
        help="Number of output tokens to optimize over"
    )
    parser.add_argument(
        "--optimizer_iter",
        type=int,
        default=10,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for finetuning"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank for finetuning"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha for finetuning"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/finetuning",
        help="Output directory for finetuning results"
    )
    args = parser.parse_args()

    return args


def preprocess_function(examples: dict, tokenizer: AutoTokenizer):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for i in range(len(examples['input_text'])):
        input_prompt = examples['input_text'][i]
        output_ids = examples['target_token_ids'][i]

        template = [{"role": "user", "content": input_prompt}]

        tokenized_prompt = tokenizer.apply_chat_template(
            template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        input_prompt_ids = tokenized_prompt.input_ids[0].squeeze(0).tolist()

        full_ids = input_prompt_ids + output_ids

        labels = ([-100] * len(input_prompt_ids)) + list(output_ids)

        attention_mask = [1] * len(full_ids)

        batch_input_ids.append(full_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }


def main(
    model_name: str,
    quantize_4bit: bool,
    quantize_8bit: bool,
    system_prompt: str | None,
    style: str | None,
    dataset_size: int,
    dataset_name: str,
    task_hints: bool,
    batch_size: int,
    output_token_count: int,
    optimizer_iter: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    seed: int,
    output_dir: str
):
    """
    Main function for finetuning

    Args:
        model_name (str) - Huggingface model name or path to use for optimization.
        quantize_4bit (bool) - If True, 4-bit quantization is enabled for the model
        quantize_8bit (bool) - If True, 8-bit quantization is enabled for the model
        system_prompt (str | None) - Custom system prompt to use for optimization.
        style (str | None) - Predefined style for the system prompt.
        dataset_size (int) - Dataset size for optimization (80:20 split)
        dataset_name (str) - Dataset to use for optimization
        task_hints (bool) - Whether to use task hints
        output_token_count (int) - Number of output tokens to finetune over
        optimizer_iter (int) - Number of finetuning iterations
        lr (float) - Learning rate for finetuning
        lora_r (int) - LoRA rank for finetuning
        lora_alpha (int) - LoRA alpha for finetuning
        seed (int) - Seed for reproducibility
        output_dir (str) - Output directory for finetuning results
    """
    logger = logging.getLogger(__name__)

    if quantize_4bit:
        quantization_mode = "4bit"
    elif quantize_8bit:
        quantization_mode = "8bit"
    else:
        quantization_mode = None
    
    set_seed(seed)
    logger.info("Loading tokenizer and model...")
    try:
        model_wrapper = Model(model_name, quantization_mode)
    except Exception as e:
        logger.exception(f"Failed to load model '{model_name}'. Error: {e}")
        return
    
    logger.info("Loading dataset...")
    try:
        train_samples, test_samples, task_system_prompt = load_and_prepare_dataset(
            dataset_name,
            dataset_size,
            task_hints,
            seed,
            split_ratio=0.8
        )
    except ValueError as e:
        logger.error(f"Error during dataset preparation: {e}")
        return
    except Exception as e:
        logger.exception(f"An unexpected error occurred during dataset preparation for {dataset_name}.")
        return
    
    logger.debug(f"Example training sample: {train_samples[0]}")

    logger.info("Constructing system prompt for finetuning...")
    if system_prompt is not None:
        conventional_sys_prompt = system_prompt
        logger.debug(f"Using custom system prompt: '{conventional_sys_prompt}'")
    elif style is not None:
        conventional_sys_prompt = get_style_prompt(style)
        if not conventional_sys_prompt:
            logger.error(f"Style '{style}' not found in predefined styles (src/style_prompts.py). Exiting.")
            sys.exit(1)
        logger.debug(f"Using style prompt for '{style}': '{conventional_sys_prompt}'")
    
    pad_token_string = model_wrapper.tokenizer.pad_token

    if not task_hints:
        conventional_sys_prompt = f"{pad_token_string}{task_system_prompt} {conventional_sys_prompt}{pad_token_string}"
    else:
        conventional_sys_prompt = f"{pad_token_string}{conventional_sys_prompt}{pad_token_string}"

    train_dataset = TextDataset(train_samples)

    conventional_collate_fn = create_collate_fn(
        tokenizer=model_wrapper.tokenizer,
        system_prompt=conventional_sys_prompt
    )

    train_dataloader_conventional = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=conventional_collate_fn,
        shuffle=False
    )


    logger.info("Precomputing model outputs (probs and IDs) using conventional system prompt...")
    precomputed_probs, precomputed_ids, max_generated_length = precompute_model_outputs(
        model_wrapper=model_wrapper,
        dataloader=train_dataloader_conventional,
        max_new_tokens=output_token_count
    )

    output_token_count = max_generated_length
    model_wrapper.tokenizer.padding_side = "right"
    precomputed_ids = precomputed_ids.transpose(0, 1)

    precomputed_ids_list = precomputed_ids.tolist()


    if hasattr(model_wrapper.model.config, "use_cache"):
        model_wrapper.model.config.use_cache = False

    data_dict = {"input_text": train_samples, "target_token_ids": precomputed_ids_list}
    hf_dataset = Dataset.from_dict(data_dict)

    logger.debug(f"Example entry from dataset: {hf_dataset[0]}")

    tokenized_dataset = hf_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size, 
        remove_columns=hf_dataset.column_names,
        fn_kwargs={"tokenizer": model_wrapper.tokenizer}
    )


    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
    )

    model_wrapper.model.train()
    peft_model = prepare_model_for_kbit_training(model_wrapper.model, use_gradient_checkpointing=True)
    peft_model = get_peft_model(peft_model, lora_config)
    logger.debug(peft_model.print_trainable_parameters())

    per_device_train_batch_size = batch_size
    learning_rate = lr
    num_train_epochs = optimizer_iter
    logging_steps = 10
    logging_strategy = "steps"
    save_strategy = "no"
    optim = "paged_adamw_8bit"

    if torch.cuda.is_bf16_supported():
        fp16_flag = False
        bf16_flag = True
        logger.info("bfloat16 is supported. Using bf16 for training.")
    else:
        fp16_flag = True
        bf16_flag = False
        logger.info("bfloat16 not supported. Using fp16 for training.")

    output_dir_path = Path(output_dir)

    output_dir_path.mkdir(parents=True, exist_ok=True)
    adapter_output_dir = output_dir_path / "lora_adapters"

    
    training_args = TrainingArguments(
        output_dir=str(adapter_output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        optim=optim,
        fp16=fp16_flag,
        bf16=bf16_flag,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        logging_first_step=True,
        save_strategy=save_strategy,
        save_total_limit=num_train_epochs,
        save_only_model=True
    )

    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=model_wrapper.tokenizer,
        mlm=False,
    )

    train_dataset_for_trainer = tokenized_dataset
    eval_dataset_for_trainer = None

    gpu_mem_cb = GpuMemoryCallbackIntegrated(log_interval_steps=training_args.logging_steps)
    manual_save_cb = ManualAdapterSaveCallback(adapter_base_save_dir=adapter_output_dir)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset_for_trainer,
        eval_dataset=eval_dataset_for_trainer,
        data_collator=data_collator,
        callbacks=[gpu_mem_cb, manual_save_cb]
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Done.")



    params = {
        "model_name": model_name,
        "quantize_4bit": quantize_4bit,
        "quantize_8bit": quantize_8bit,
        "system_prompt": conventional_sys_prompt,
        "style": style,
        "obfuscation_method": "finetuning",
        "batch_size": batch_size,
        "dataset_size": dataset_size,
        "dataset_name": dataset_name,
        "task_hints": task_hints,
        "output_token_count": output_token_count,
        "learning_rate": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "optimizer_iter": optimizer_iter,
        "seed": seed
    }

    with open(output_dir_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    prepared_data_dir = output_dir_path / "prepared_data"
    prepared_data_dir.mkdir(parents=True, exist_ok=True)

    train_texts_file = prepared_data_dir / "train_data.json"
    test_texts_file = prepared_data_dir / "test_data.json"

    with open(train_texts_file, "w") as f:
        json.dump(train_samples, f, indent=4)

    with open(test_texts_file, "w") as f:
        json.dump(test_samples, f, indent=4)

    logger.info(f"Results saved to {output_dir_path.resolve()}")

if __name__ == "__main__":
    setup_logging('finetune.log', 'DEBUG')
    logger = logging.getLogger(__name__)

    logger.debug("Parsing command line arguments...")
    try:
        args = get_args()
        logger.info(f"Command line arguments received: {json.dumps(vars(args), indent=2)}")
        main(**vars(args))
    except SystemExit:
        logger.warning("Exiting due to argument parsing issue (e.g., --help or invalid arguments).")
    except Exception as e:
        logger.exception(f"An critical error occurred: {e}")
    finally:
        logger.info("Done.")