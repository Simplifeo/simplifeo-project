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
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# --- CONFIGURATION ---
DATASET_PATH = "data/questions_dataset.json"
BASE_MODEL_ID = "google/pix2struct-base"
OUTPUT_MODEL_DIR = "simplifeo-bank-statement-model"

# --- 1. Préparation du Jeu de Données ---
class BankStatementDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, processor):
        self.dataset = json.load(open(dataset_path))
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = os.path.join(os.getcwd(), item['image_path'])
        image = Image.open(image_path)
        question = item['question']
        answer = item['answer']

        encoding = self.processor(
            images=image,
            text=f"Question: {question} Answer: {answer}",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding

# --- 2. Fonction Principale d'Entraînement ---
def train():
    print("Début du processus de fine-tuning...")

    processor = Pix2StructProcessor.from_pretrained(BASE_MODEL_ID)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = Pix2StructForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "value"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = BankStatementDataset(dataset_path=DATASET_PATH, processor=processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        logging_steps=50,
        save_steps=500,
        learning_rate=2e-4,
        remove_unused_columns=False,
        fp16=True,
        gradient_checkpointing=True,  # LA CORRECTION FINALE
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
    train()