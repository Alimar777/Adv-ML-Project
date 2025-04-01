outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=60,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )