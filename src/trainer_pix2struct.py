# src/trainer_pix2struct.py

import os
import json
import torch
from PIL import Image
from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig  # On importe la configuration explicite
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
        # On s'assure que le chemin est bien relatif au dossier du projet
        image_path = os.path.join(os.getcwd(), item['image_path'])
        image = Image.open(image_path)
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

    # --- NOUVELLE SECTION CORRIGÉE ---
    # Configuration de quantification explicite pour éviter les conflits de type
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # On force le type de calcul pour être compatible
    )

    # Charger le modèle avec notre configuration précise
    model = Pix2StructForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    
    # Désactiver le cache qui peut causer des problèmes lors de l'entraînement
    model.config.use_cache = False
    # --- FIN DE LA NOUVELLE SECTION ---

    # Préparer le modèle pour l'entraînement avec PEFT/LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "value"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Charger notre jeu de données
    dataset = BankStatementDataset(dataset_path=DATASET_PATH, processor=processor)

    # Définir les arguments de l'entraînement
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        logging_steps=50,
        save_steps=500,
        learning_rate=2e-4,
        remove_unused_columns=False,
        fp16=True,  # On garde cette ligne, elle est cruciale
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
    final_checkpoint_dir = os.path.join(OUTPUT_MODEL_DIR, "final_checkpoint")
    model.save_pretrained(final_checkpoint_dir)
    print(f"✔ Modèle entraîné sauvegardé dans '{final_checkpoint_dir}'")


if __name__ == "__main__":
    train()