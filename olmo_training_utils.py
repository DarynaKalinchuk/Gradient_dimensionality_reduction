import torch

SFT_MESSAGE_KEY = "messages"
INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"

def sft_tulu_tokenize_and_truncate_v1(row, tokenizer, max_seq_length=4096):
    """Taken more or less directly from https://github.com/allenai/open-instruct/blob/main/open_instruct/finetune.py
       (who took it directly from https://github.com/allenai/open-instruct/blob/ba11286e5b9eb00d4ce5b40ef4cac1389888416a/open_instruct/finetune.py#L385)"""

    tokenizer.chat_template = """{% for message in messages %}
    {{ '<|user|>' if message['role'] == 'user' else '<|assistant|>' }} {{ message['content'].strip() }}
    {% endfor %}"""

    messages = row["messages"]

    # Flatten if necessary
    if isinstance(messages, list) and len(messages) == 1 and isinstance(messages[0], list):
        messages = messages[0]  # minus one level of nesting

    # Validate structure
    if not (isinstance(messages, list) and all(isinstance(m, dict) for m in messages)):
        raise ValueError(f"Invalid message format: {messages}")

    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    # print(messages)
    # first tokenize without padding/truncation to calculate positions
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
        truncation=False,  # avoids truncating here
        padding=False,     # no padding here yet
        add_generation_prompt=False,
    )

    # print(input_ids)

    # Copy input_ids for labels
    labels = input_ids.clone()

    current_pos = 0
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],  # Up to the previous message
                    tokenize=True,
                    return_tensors="pt",
                    truncation=False, 
                    padding=False,  #was: return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=4096
                    add_generation_prompt=False,
                ).shape[1]

            # next, we calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    truncation=False,
                    padding=False, # had to disable padding and truncation here as well
                    add_generation_prompt=True,
                ).shape[1]
            else:
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    truncation=False,
                    padding=False, # was: return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=4096
                    add_generation_prompt=False,
                ).shape[1]

            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100

            if max_seq_length and message_end_idx >= max_seq_length:
                break

    # Now padding and truncation. When used earlier, results in 0 gradients
    if input_ids.shape[1] > max_seq_length:
        input_ids = input_ids[:, :max_seq_length]
        labels = labels[:, :max_seq_length]
        attention_mask = torch.ones_like(input_ids)
    else:
        
        padding_length = max_seq_length - input_ids.shape[1]
        input_ids = torch.cat([
            input_ids,
            torch.full((1, padding_length), tokenizer.pad_token_id, dtype=torch.long)
        ], dim=1)
        labels = torch.cat([
            labels,
            torch.full((1, padding_length), -100, dtype=torch.long)
        ], dim=1)
        attention_mask = torch.cat([
            torch.ones_like(input_ids[0, :-padding_length]),
            torch.zeros_like(input_ids[0, -padding_length:])
        ], dim=0).unsqueeze(0)

    # no .flatten()
    row[INPUT_IDS_KEY] = input_ids
    row[LABELS_KEY] = labels
    row[ATTENTION_MASK_KEY] = attention_mask

    return row
