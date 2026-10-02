"""Write the SFT starting checkpoint: Qwen3-1.7B-Base with a usable <|im_end|> row.

The base model barely trained <|im_end|>; its embedding norm is ~0.4 against ~1.5-1.9
for ordinary tokens. Embeddings are tied to the LM head, so the token's logit stays near
zero and SFT cannot raise it: sampled turns run past the end instead of stopping.
Swapping the <|im_end|> and <|endoftext|> rows fixes the starting point: <|im_end|>
gets a trained stop-token row, and <|endoftext|>, which the chat format never uses,
gets the weak one, so the two do not compete at the end of each turn.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-1.7B-Base"
out = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
assert model.config.tie_word_embeddings
embed = model.get_input_embeddings().weight
im_end, endoftext = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
with torch.no_grad():
    embed[[im_end, endoftext]] = embed[[endoftext, im_end]]
model.save_pretrained(out)
tokenizer.save_pretrained(out)
print(f"wrote {out}")
