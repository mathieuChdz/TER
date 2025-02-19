from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

chat = [
   {"role": "user", "content": "Who are you ?"},
]
inputs = tokenizer.apply_chat_template(chat, return_tensors="pt")
inputs = inputs.to("cuda")

output = model.generate(input_ids=inputs, max_new_tokens=200)
output = output[0].to("cpu")
print(tokenizer.decode(output))