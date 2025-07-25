# src/trainer_pix2struct.py

import os
import json
import torch
from PIL import Image
from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# --- CONFIGURATION ---
DATASET_PATH = "data/questions_dataset.json"
BASE_MODEL_ID = "google/pix2struct-base"
OUTPUT_MODEL_DIR = "simplifeo-bank-statement-model" # Le dossier où sera sauvegardé notre modèle entraîné

# --- 1. Préparation du Jeu de Données ---
class BankStatementDataset(torch.utils.data.Dataset):
    """
    Classe personnalisée pour charger notre jeu de données de questions/réponses.
    Elle transforme nos données en un format que le Trainer de Hugging Face peut comprendre.
    """
    def __init__(self, dataset_path, processor):
        self.dataset = json.load(open(dataset_path))
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(item['image_path'])
        question = item['question']
        answer = item['answer']

        # Le processeur prépare l'image et le texte pour le modèle
        encoding = self.processor(
            images=image,
            text=f"Question: {question} Answer: {answer}",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        # On doit "aplatir" les tenseurs pour que le modèle les accepte
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding

# --- 2. Fonction Principale d'Entraînement ---
def train():
    print("Début du processus de fine-tuning...")

    # Charger le processeur
    processor = Pix2StructProcessor.from_pretrained(BASE_MODEL_ID)

    # Charger le modèle avec la technique QLoRA pour économiser la mémoire
    # - `load_in_4bit=True`: Charge le modèle en utilisant des nombres de 4 bits (très léger)
    # - `torch_dtype=torch.bfloat16`: Utilise un type de nombre optimisé pour les GPU modernes
    model = Pix2StructForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
    )
    
    model.config.use_cache = False

    # Préparer le modèle pour l'entraînement avec PEFT/LoRA
    # On ne va entraîner que de petites "couches d'adaptation" (environ 1% du modèle)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "value"] # On cible des parties spécifiques du modèle
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # Affiche le % de paramètres que nous entraînons réellement

    # Charger notre jeu de données
    dataset = BankStatementDataset(dataset_path=DATASET_PATH, processor=processor)

    # Définir les arguments de l'entraînement
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=3,  # On va montrer le jeu de données 3 fois au modèle
        per_device_train_batch_size=2, # On traite les images 2 par 2
        logging_steps=50, # Affiche la progression tous les 50 pas
        save_steps=500, # Sauvegarde un checkpoint tous les 500 pas
        learning_rate=2e-4,
        remove_unused_columns=False,
        fp16=True, 
    )

    # Créer l'objet Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("\n--- Lancement de l'entraînement ---")
    # Lancer le fine-tuning !
    trainer.train()
    print("--- Entraînement terminé ---")

    # Sauvegarder le modèle final (uniquement les couches LoRA entraînées)
    model.save_pretrained(os.path.join(OUTPUT_MODEL_DIR, "final_checkpoint"))
    print(f"✔ Modèle entraîné sauvegardé dans '{OUTPUT_MODEL_DIR}/final_checkpoint'")


if __name__ == "__main__":
    # Ce bloc permet d'exécuter la fonction train() si on lance le script
    # depuis la ligne de commande.
    train()