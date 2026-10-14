"""Chat-template encoding that supervises assistant spans only."""


def encode_messages(tokenizer, messages: list[dict], tools: list | None,
                    final_only: bool = False) -> dict:
    """Mask all system/user/tool tokens; supervise complete assistant turns.

    ``final_only`` supervises just the last assistant turn. Managed agent
    contexts rebuild the state summary before every call, so earlier actions in
    a later prefix are conditioned on their own results and must stay masked.
    """
    options = {'tools': tools, 'enable_thinking': False, 'tokenize': False}
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
    spans = []
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prefix = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, **options)
        ending = tokenizer.apply_chat_template(messages[:i+1], add_generation_prompt=False, **options)
        if not full.startswith(prefix) or not full.startswith(ending):
            raise ValueError('assistant template boundary mismatch')
        spans.append((len(prefix), len(ending)))
    if final_only:
        spans = spans[-1:]
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    labels = [token if any(start >= a and end <= b and end > start for a, b in spans) else -100
              for token, (start, end) in zip(encoded['input_ids'], encoded['offset_mapping'])]
    if not any(label != -100 for label in labels):
        raise ValueError('empty assistant supervision')
    return {'input_ids': encoded['input_ids'], 'labels': labels}
