from transformers import (
    AutoModelForSeq2SeqLM,
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

print("###fresh model init###")

model = T5ForConditionalGeneration.from_pretrained("/home/student/f/fklinge/share/bachelor/ichteste/minillm/checkpoints/flan-t5-large")
tokenizer = T5Tokenizer.from_pretrained("/home/student/f/fklinge/share/bachelor/ichteste/minillm/checkpoints/flan-t5-large")
tokenizer.save_pretrained("/home/student/f/fklinge/share/bachelor/ichteste/minillm/checkpoints/flan-t5-large-sft")
model.save_pretrained("/home/student/f/fklinge/share/bachelor/ichteste/minillm/checkpoints/flan-t5-large-sft")

print("###fresh model saved###")

# # Prepare some sample input
# input_text = "Translate English to French: Hello, how are you?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# output_text = "Bonjour, comment ça va?"
# labels = tokenizer(output_text, return_tensors="pt").input_ids

# # Perform a forward pass  
# logits = model(input_ids=input_ids, labels=labels).logits


# print("***logits for dummy input", logits, "\n\n")
