from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import re

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

def use_model(model, tokenizer):
    
    prompt = "Ecrit un rapport détaillé sur le sujet suivant : " \
             " 'GPU et outils de développement" \
             " Les GPU sont de plus en plus utilisés pour accélérer les calculs dans les applications." \
             " L’objectif de ce projet est d’étudier les outils et les techniques mis en œuvre pour développer/monitorer ce type de processeurs." \
             " Les tâches à réaliser sont donc :" \
                " - étudier les environnements disponibles pour le développement sur GPU " \
                " - mettre en place un environnement de test " \
                " - implémenter quelques algorithmes et mesurer leur efficacité ' " \
            " De plus, voici ce qui doit aussi apparaître dans le rapport : " \
            "   - Quoi faire sur GPU" \
            "   - Comment programmer sur GPU (langages, ect)" \
            "   - Bibliothèques python" \
            "   - Machine Learning" \
            "   - Comparaison CPU vs GPU" \
            "Ce rapport doit être écrit en markdown. Ajoute un sommaire avec titre et des sous partie si nécéssaire."
    
              

    chat = [{"role": "user", "content": prompt},]
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt")
    inputs = inputs.to("cuda")
    output = model.generate(input_ids=inputs, max_new_tokens=100000)
    output = output[0].to("cpu")
    response = tokenizer.decode(output, skip_special_tokens=True)
    response = re.sub(r'\[INST\].*?\[\/INST\]', '', response).strip()
    with open("rapport_ia.md", "w", encoding="utf-8") as file:
        file.write(response)
    print(f"Model: {response}\n")


if __name__ == "__main__":
    
    model_id = "codellama/CodeLlama-7b-Instruct-hf"

    model, tokenizer = load_model(model_id)
    try :
        use_model(model, tokenizer)
    finally :
        unload_model(model, tokenizer)
