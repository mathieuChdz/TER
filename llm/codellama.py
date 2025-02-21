from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
   model_id,
   torch_dtype=torch.float16,
   device_map="auto",
)

chat = [
   {"role": "user", "content": "Who are you ?"},
]
inputs = tokenizer.apply_chat_template(chat, return_tensors="pt")
inputs = inputs.to("cuda")

output = model.generate(input_ids=inputs, max_new_tokens=200)
output = output[0].to("cpu")
print(tokenizer.decode(output))