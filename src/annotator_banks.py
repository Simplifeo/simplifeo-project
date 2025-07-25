# src/annotator_banks.py

import os
import csv
import json
import random
import pandas as pd # On utilise pandas pour lire le CSV plus facilement

# --- CONFIGURATION ---
INPUT_CSV = "data/synthetic_bank_statements/ground_truth.csv"
OUTPUT_JSON = "data/questions_dataset.json"
TOTAL_QUESTIONS_TO_GENERATE = 500 # On peut ajuster ce nombre

def generate_questions_for_row(row):
    """Génère une liste de paires (question, réponse) pour une ligne du CSV."""
    
    questions = []
    
    # Questions sur les champs simples
    questions.append({
        "question": random.choice(["Quel est le nom de la banque ?", "Nom de l'établissement bancaire ?", "Quelle est la banque ?"]),
        "answer": row['bank_name']
    })
    questions.append({
        "question": random.choice(["Quel est le nouveau solde ?", "Solde final ?", "Montant du nouveau solde ?"]),
        "answer": f"{row['end_balance']} €"
    })
    questions.append({
        "question": random.choice(["Quel est le titulaire du compte ?", "Nom du client ?"]),
        "answer": row['account_holder']
    })
    
    # Questions sur les transactions (la partie la plus importante)
    # On doit "lire" la chaîne de caractères JSON pour la transformer en vraie liste
    transactions = json.loads(row['transactions_json'].replace("'", '"'))
    
    # On choisit une transaction au hasard pour poser une question dessus
    if transactions:
        transaction = random.choice(transactions)
        
        # Si c'est un débit
        if transaction['debit']:
            questions.append({
                "question": f"Quel est le montant du débit pour l'opération '{transaction['description']}' ?",
                "answer": f"{transaction['debit']} €"
            })
        # Si c'est un crédit
        if transaction['credit']:
            questions.append({
                "question": f"Quel est le montant du crédit pour l'opération '{transaction['description']}' ?",
                "answer": f"{transaction['credit']} €"
            })

    return questions

def main():
    """Fonction principale pour créer le jeu de données d'entraînement."""
    print(f"Début de la création du jeu de données depuis '{INPUT_CSV}'...")
    
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier '{INPUT_CSV}' n'a pas été trouvé. Assurez-vous d'avoir lancé la data_factory.")
        return

    final_dataset = []
    
    # On boucle jusqu'à avoir assez de questions
    while len(final_dataset) < TOTAL_QUESTIONS_TO_GENERATE:
        # On choisit une ligne au hasard dans notre CSV
        random_row = df.sample(n=1).iloc[0]
        
        # On génère les questions pour cette ligne
        generated_qs = generate_questions_for_row(random_row)
        
        # On ajoute l'image associée à chaque question
        for q in generated_qs:
            q['image_path'] = os.path.join("data/synthetic_bank_statements", random_row['file_name'])
            final_dataset.append(q)

    print(f"Génération de {len(final_dataset)} paires de questions/réponses.")

    # Sauvegarder le jeu de données final dans un fichier JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"✔ Terminé ! Jeu de données sauvegardé dans '{OUTPUT_JSON}'.")


if __name__ == "__main__":
    main()