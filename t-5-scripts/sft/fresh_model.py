import sys
import os

from transformers import (
    AutoModelForSeq2SeqLM,
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

print("###fresh model init###")

base_path = sys.argv[1]
ckpt = sys.argv[2]

model = T5ForConditionalGeneration.from_pretrained(os.path.join(base_path, "checkpoints", ckpt))
tokenizer = T5Tokenizer.from_pretrained(os.path.join(base_path, "checkpoints", ckpt))
tokenizer.save_pretrained(os.path.join(base_path, "checkpoints", ckpt + "-new"))
model.save_pretrained(os.path.join(base_path, "checkpoints", ckpt + "-new"))

print("###fresh model saved###")

# # Prepare some sample input
# input_text = "Translate English to French: Hello, how are you?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# output_text = "Bonjour, comment ça va?"
# labels = tokenizer(output_text, return_tensors="pt").input_ids

# # Perform a forward pass  
# logits = model(input_ids=input_ids, labels=labels).logits


# print("***logits for dummy input", logits, "\n\n")
