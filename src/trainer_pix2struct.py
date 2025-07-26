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
import argparse

# --- CONFIGURATION ---
DATASET_PATH = "data/questions_dataset.json"
BASE_MODEL_ID = "google/pix2struct-base"
OUTPUT_MODEL_DIR = "simplifeo-bank-statement-model"

# --- 1. Préparation du Jeu de Données ---
class BankStatementDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, processor):
        self.dataset_path = dataset_path
        self.dataset = json.load(open(dataset_path))
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        print(f"\rPréparation des données... ({idx + 1}/{self.__len__()})", end="")
        
        item = self.dataset[idx]
        # --- CORRECTION FINALE DU CHEMIN ---
        # On utilise directement le chemin stocké dans le JSON, qui est déjà correct.
        image_path = item['image_path']
        
        try:
            image = Image.open(image_path)
            question = item['question']
            answer = item['answer']

            inputs = self.processor(
                images=image,
                text=f"Question: {question}",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )

            labels = self.processor(
                text=answer,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            ).input_ids

            inputs['labels'] = labels
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            print(f"\n\n--- ERREUR ---")
            print(f"Impossible de traiter l'exemple #{idx} (image: {item['image_path']})")
            print(f"Raison de l'erreur : {e}")
            print(f"Cet exemple sera ignoré.\n")
            return None

# --- 2. Fonction Principale d'Entraînement ---
def train(smoke_test=False):
    print("Début du processus de fine-tuning...")

    processor = Pix2StructProcessor.from_pretrained(BASE_MODEL_ID)
    dataset = BankStatementDataset(dataset_path=DATASET_PATH, processor=processor)

    if smoke_test:
        print("\n--- LANCEMENT DU SMOKE TEST LOCAL ---")
        print("Vérification de la préparation de 5 exemples...")
        for i in range(5):
            _ = dataset[i]
        print("\n\n✔ Smoke test terminé avec succès ! Le code de préparation des données est valide.")
        return

    model = Pix2StructForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
    )
    
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=50,
        save_steps=500,
        learning_rate=2e-4,
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("\n--- Lancement de l'entraînement ---")
    trainer.train()
    print("--- Entraînement terminé ---")

    final_checkpoint_dir = os.path.join(OUTPUT_MODEL_DIR, "final_checkpoint")
    model.save_pretrained(final_checkpoint_dir)
    print(f"✔ Modèle entraîné sauvegardé dans '{final_checkpoint_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Lance une vérification rapide sans entraînement'
    )
    args = parser.parse_args()

    train(smoke_test=args.smoke_test)