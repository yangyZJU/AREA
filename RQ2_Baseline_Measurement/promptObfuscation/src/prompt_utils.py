import logging
import torch
from typing import List, Tuple
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# Dictionary mapping model names to tuples of beginning and end token IDs for system prompts. We need to hardcode this, since this has to work with blank system prompts as well
MODEL_TEMPLATE_SYS_PROMPT_BEGINNING_END_IDS = {

    "meta-llama/Meta-Llama-3.1-8B-Instruct": (271, 128009),
    "meta-llama/Llama-2-7b-chat-hf": (13, 13),
    "tiiuae/Falcon3-7B-Instruct": (12, 12),
    "Qwen/Qwen3-30B-A3B-Instruct-2507":(198, 151645),
}

def apply_chat_template_to_batch(
    user_prompts: List[str],
    system_prompt: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[str]:
    """
    Applies the tokenizer's chat template to a batch of user prompts
    with a given system prompt.

    Args:
        user_prompts: A list of user input strings.
        system_prompt: The content for the system role.
        tokenizer: The tokenizer to use.

    Returns:
        A list of formatted strings ready for tokenization.
    """
    formatted_prompts = []
    for user_prompt in user_prompts:
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({'role': 'user', 'content': user_prompt})
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        )
    return formatted_prompts


def zero_pad_token_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int
) -> torch.Tensor:
    """
    Zeros out the attention mask for all occurrences of the pad_token_id.

    Args:
        input_ids: Tensor of input IDs. (Batch_size, Seq_len)
        attention_mask: Tensor of attention_masks. (Batch_size, Seq_len)
        pad_token_id: The ID of the padding token.

    Returns:
        A new attention_mask tensor with pad token positions zeroed out.
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError(f"input_ids shape {input_ids.shape} and attention_mask shape {attention_mask.shape} must match.")
    
    new_attention_mask = attention_mask.clone()
    pad_token_positions = (input_ids == pad_token_id)
    new_attention_mask[pad_token_positions] = 0
    return new_attention_mask


def generate_random_token_sequence(
    num_tokens: int,
    vocab_size: int
) -> torch.Tensor:
    """
    Generates a random sequence of token IDs.

    Args:
        num_tokens: The length of the token sequence.
        vocab_size: The total number of tokens in the tokenizer's vocabulary.

    Returns:
        A 1D tensor of random token IDs.
    """
    return torch.randint(0, vocab_size, (num_tokens,))


def _extract_delimited_prompt_segment(
    prompt_with_delimiters: torch.Tensor,
    delimiter_token_id: int
) -> torch.Tensor:
    """
    Extracts the segment of a prompt that is enclosed by delimiter tokens,
    including the delimiter tokens themselves.
    It expects exactly two delimiter tokens.

    Args:
        prompt_with_delimiters: 1D Tensor containing the prompt IDs,
                                e.g., [..., DELIMITER, content1, ..., contentN, DELIMITER, ...].
        delimiter_token_id: The ID of the delimiter token.

    Returns:
        torch.Tensor: The 1D segment of the prompt including the two delimiter tokens,
                      e.g., [DELIMITER, content1, ..., contentN, DELIMITER].

    Raises:
        TypeError: If prompt_with_delimiters is not a torch.Tensor.
        ValueError: If prompt_with_delimiters is not 1D or does not contain exactly two delimiter_token_id tokens.
    """
    if not isinstance(prompt_with_delimiters, torch.Tensor):
        raise TypeError("prompt_with_delimiters must be a torch.Tensor.")
    if not prompt_with_delimiters.ndim == 1:
        raise ValueError("prompt_with_delimiters must be a 1D tensor.")

    delimiter_positions = torch.nonzero(prompt_with_delimiters == delimiter_token_id, as_tuple=False).squeeze(-1)

    if delimiter_positions.numel() != 2:
        raise ValueError(
            f"Input tensor for _extract_delimited_prompt_segment must contain exactly two "
            f"delimiter_token_id ({delimiter_token_id}) tokens. Found {delimiter_positions.numel()} "
            f"at positions {delimiter_positions.tolist()} in tensor of size {prompt_with_delimiters.size(0)}."
        )

    start_idx = delimiter_positions[0].item()
    end_idx = delimiter_positions[1].item()

    return prompt_with_delimiters[start_idx : end_idx + 1]

def find_sys_prompt_indices(
    input_ids: torch.Tensor,
    sys_prompt_ids: torch.Tensor,
    pad_token_id: int,
    model_name: str
) -> Tuple:
    """
    Finds the start and end indices of the system prompt in a given input tensor.

    Args:
        input_ids: A 1D tensor of token IDs.
        sys_prompt_ids: A 1D tensor of token IDs that encodes the system prompt.
        pad_token_id: The ID of the padding token.
        model_name: The name of the model.

    Returns:
        A tuple containing the start and end indices of the system prompt in the input tensor.
    """
    sys_prompt_ids = _extract_delimited_prompt_segment(sys_prompt_ids, pad_token_id)

    template_beginning_id, template_end_id = MODEL_TEMPLATE_SYS_PROMPT_BEGINNING_END_IDS[model_name]
    first_id_tensor = torch.tensor([template_beginning_id], dtype=sys_prompt_ids.dtype, device=sys_prompt_ids.device)
    second_id_tensor = torch.tensor([template_end_id], dtype=sys_prompt_ids.dtype, device=sys_prompt_ids.device)
    sys_prompt_ids = torch.cat([first_id_tensor, sys_prompt_ids, second_id_tensor])
    
    input_len = input_ids.size(0)
    sys_prompt_len = sys_prompt_ids.size(0)

    for i in range(input_len - sys_prompt_len + 1):
        sub_tensor = input_ids[i:i + sys_prompt_len]
        if(torch.equal(sub_tensor, sys_prompt_ids)):
            start_index = i
            end_index = i + sys_prompt_len - 1
            return (start_index+2, end_index-1)

    raise ValueError("System prompt not found in input tensor.")


def find_sys_prompt_indices_batch(
    input_ids: torch.Tensor,
    sys_prompt_ids: torch.Tensor,
    pad_token_id: int,
    model_name
) -> Tuple:
    indices = [find_sys_prompt_indices(input_id, sys_prompt_ids, pad_token_id, model_name) for input_id in input_ids]
    return indices


def update_attention_mask(
    obf_sys_prompt_len: int,
    attention_mask: torch.Tensor,
    sys_prompt_indices: Tuple
) -> torch.Tensor:
    """
    Updates the attention mask to account for the obfuscated system prompt.

    Args:
        obf_sys_prompt_len (int): The length of the obfuscated system prompt.
        attention_mask (torch.Tensor): The attention mask tensor.
        sys_prompt_indices (Tuple): A tuple containing the start and end indices of the system prompt.

    Returns:
        torch.Tensor: The updated attention mask tensor.
    """
    start_index, end_index = sys_prompt_indices
    return torch.cat((attention_mask[:start_index], torch.ones(obf_sys_prompt_len), attention_mask[end_index:]), dim=0)

def update_attention_mask_batch(
    obf_sys_prompt_len: int,
    attention_masks: torch.Tensor,
    sys_prompt_indices: List[Tuple]
) -> torch.Tensor:
    new_attention_masks = [update_attention_mask(obf_sys_prompt_len, attention_mask, sys_prompt_indices[idx]) for idx, attention_mask in enumerate(attention_masks)]
    return torch.stack(new_attention_masks, dim=0)


def replace_sys_prompt(
    new_sys_prompt: torch.Tensor,
    prompt: torch.Tensor,
    sys_prompt_indices: Tuple
) -> torch.Tensor:
    """
    Replaces the system prompt in a tensor with a new system prompt.

    Args:
        new_sys_prompt (torch.Tensor): The new system prompt tensor.
        prompt (torch.Tensor): The prompt tensor.
        sys_prompt_indices (Tuple): A tuple containing the start and end indices of the system prompt.

    Returns:
        torch.Tensor: The prompt tensor with the system prompt replaced by the new system prompt.
    """
    start_index, end_index = sys_prompt_indices
    return torch.cat((prompt[:start_index], new_sys_prompt, prompt[end_index:]), dim=0)
    

def replace_sys_prompt_batch(
    new_sys_prompt: torch.Tensor,
    prompts: torch.Tensor,
    sys_prompt_indices: List[Tuple]
) -> torch.Tensor:
    new_prompts = [replace_sys_prompt(new_sys_prompt, prompt, sys_prompt_indices[idx]) for idx, prompt in enumerate(prompts)]
    return torch.stack(new_prompts, dim=0)