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

ckpt_student = sys.argv[2]
ckpt_teacher = sys.argv[3]


student = T5ForConditionalGeneration.from_pretrained(os.path.join(base_path, "checkpoints", ckpt_student))
teacher = T5ForConditionalGeneration.from_pretrained(os.path.join(base_path, "checkpoints", ckpt_teacher))
# Using same tokenizer for student and teacher
tokenizer = T5Tokenizer.from_pretrained(os.path.join(base_path, "checkpoints", ckpt_teacher))
tokenizer.save_pretrained(os.path.join(base_path, "checkpoints", ckpt_student + "-minillm"))
tokenizer.save_pretrained(os.path.join(base_path, "checkpoints", ckpt_teacher + "-minillm"))
student.save_pretrained(os.path.join(base_path, "checkpoints", ckpt_student + "-minillm"))
teacher.save_pretrained(os.path.join(base_path, "checkpoints", ckpt_student + "-minillm"))

print("###fresh model saved###")

