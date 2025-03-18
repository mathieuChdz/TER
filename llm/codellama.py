from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import re

def ask_model():
    models = {
        "1": "codellama/CodeLlama-7b-Instruct-hf",
        "2": "codellama/CodeLlama-70b-Instruct-hf"
    }
    print("Available models:")
    for key, value in models.items():
        print(f"{key}. {value.split('/')[-1]}")
    print("")
    model_number = input("Enter the model number: ")
    model_id = models.get(model_number)
    if model_id is None:
        print("Invalid model ID.")
        return ask_model()
    print("")
    return model_id

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("")
    return model, tokenizer

def unload_model(model, tokenizer):
    del model
    del tokenizer
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

def chat_with_model(model, tokenizer):
    print("Type 'exit' or 'quit' or 'q' to stop chatting.\n")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        print("")
        chat = [{"role": "user", "content": user_input},]
        inputs = tokenizer.apply_chat_template(chat, return_tensors="pt")
        inputs = inputs.to("cuda")
        output = model.generate(input_ids=inputs, max_new_tokens=100000)
        output = output[0].to("cpu")
        response = tokenizer.decode(output, skip_special_tokens=True)
        response = re.sub(r'\[INST\].*?\[\/INST\]', '', response).strip()
        print(f"Model: {response}\n")


if __name__ == "__main__":
    model_id = ask_model()
    model, tokenizer = load_model(model_id)
    try :
        chat_with_model(model, tokenizer)
    finally :
        unload_model(model, tokenizer)
