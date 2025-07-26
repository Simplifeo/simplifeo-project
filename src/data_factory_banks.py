# src/data_factory_banks.py

import os
import csv
import random
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
TOTAL_STATEMENTS = 100  # Le nombre d'images à générer
OUTPUT_DIR = "data/synthetic_bank_statements" # Le dossier de sortie

# Initialiser le générateur de fausses données en français
fake = Faker('fr_FR')

def generate_statement_data():
    """Génère les données textuelles pour un relevé bancaire complet."""
    
    # --- LOGIQUE CORRIGÉE ---
    # On s'assure que la date de début est toujours avant la date de fin.
    random_date_in_month = fake.date_object()
    start_date = random_date_in_month.replace(day=1)
    end_date = random_date_in_month.replace(day=28) # On garde 28 pour éviter les problèmes de mois courts
    
    # Informations générales du compte
    account_holder = fake.name()
    bank_name = random.choice(["BNP Paribas", "Société Générale", "Crédit Agricole", "La Banque Postale", "Crédit Mutuel"])
    iban = fake.iban()
    start_balance = round(random.uniform(500.0, 5000.0), 2)
    
    transactions = []
    current_balance = start_balance
    
    # Générer entre 5 et 15 transactions
    for _ in range(random.randint(5, 15)):
        is_debit = random.choice([True, True, False]) # 2/3 de chances d'avoir un débit
        amount = round(random.uniform(10.0, 450.0), 2)
        
        if is_debit:
            description = random.choice(["PAIEMENT CB", "PRELEVEMENT", "RETRAIT DAB", "VIREMENT A"])
            description += " " + fake.company().upper()
            debit = f"{amount:.2f}"
            credit = ""
            current_balance -= amount
        else:
            description = random.choice(["VIREMENT DE", "REMISE DE CHEQUE"])
            description += " " + fake.company().upper()
            debit = ""
            credit = f"{amount:.2f}"
            current_balance += amount
            
        transactions.append({
            "date": fake.date_between(start_date=start_date, end_date=end_date).strftime('%d/%m/%Y'),
            "description": description,
            "debit": debit,
            "credit": credit
        })
        
    end_balance = round(current_balance, 2)
    
    return {
        "account_holder": account_holder,
        "bank_name": bank_name,
        "iban": iban,
        "period": f"du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}",
        "start_balance": f"{start_balance:.2f}",
        "end_balance": f"{end_balance:.2f}",
        "transactions": transactions
    }



def create_statement_image(data, output_path):
    """Crée une image PNG à partir des données d'un relevé."""
    
    width, height = 800, 1100
    background_color = "white"
    
    # Créer une image blanche
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)
    
    # Charger une police de caractères (vous devrez peut-être ajuster le chemin)
    try:
        # Sur macOS, les polices système sont souvent ici
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        font_regular = ImageFont.truetype(font_path, 14)
        font_bold = ImageFont.truetype(font_path.replace("Arial", "Arial Bold"), 16)
        font_title = ImageFont.truetype(font_path.replace("Arial", "Arial Bold"), 24)
    except IOError:
        print("Police Arial non trouvée, utilisation de la police par défaut.")
        font_regular = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_title = ImageFont.load_default()

    # --- DESSINER LE CONTENU ---
    y_pos = 40
    
    # Titre
    draw.text((40, y_pos), "RELEVÉ DE COMPTE", fill="black", font=font_title)
    y_pos += 60
    
    # Infos générales
    draw.text((40, y_pos), data['bank_name'], fill="black", font=font_bold)
    draw.text((500, y_pos), f"Titulaire : {data['account_holder']}", fill="black", font=font_regular)
    y_pos += 20
    draw.text((500, y_pos), f"IBAN : {data['iban']}", fill="black", font=font_regular)
    y_pos += 20
    draw.text((500, y_pos), f"Période : {data['period']}", fill="black", font=font_regular)
    y_pos += 60
    
    # En-têtes du tableau
    draw.line([(40, y_pos), (width - 40, y_pos)], fill="black", width=2)
    y_pos += 10
    draw.text((50, y_pos), "Date", fill="black", font=font_bold)
    draw.text((150, y_pos), "Libellé de l'opération", fill="black", font=font_bold)
    draw.text((550, y_pos), "Débit (€)", fill="black", font=font_bold)
    draw.text((680, y_pos), "Crédit (€)", fill="black", font=font_bold)
    y_pos += 10
    draw.line([(40, y_pos), (width - 40, y_pos)], fill="black", width=2)
    y_pos += 15

    # Lignes de transactions
    for trans in data['transactions']:
        draw.text((50, y_pos), trans['date'], fill="black", font=font_regular)
        draw.text((150, y_pos), trans['description'], fill="black", font=font_regular)
        draw.text((550, y_pos), trans['debit'], fill="black", font=font_regular)
        draw.text((680, y_pos), trans['credit'], fill="black", font=font_regular)
        y_pos += 25

    # Soldes
    y_pos += 40
    draw.text((450, y_pos), f"Solde précédent : {data['start_balance']} €", fill="black", font=font_regular)
    y_pos += 25
    draw.text((450, y_pos), f"Nouveau solde : {data['end_balance']} €", fill="black", font=font_bold)

    # Sauvegarder l'image
    image.save(output_path)


def main():
    """Fonction principale pour générer les images et le fichier CSV."""
    print(f"Début de la génération de {TOTAL_STATEMENTS} relevés bancaires synthétiques...")
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_file_path = os.path.join(OUTPUT_DIR, "ground_truth.csv")
    
    # Définir les en-têtes pour notre fichier de vérité
    csv_headers = ["file_name", "account_holder", "bank_name", "iban", "period", "start_balance", "end_balance", "transactions_json"]

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        
        for i in range(TOTAL_STATEMENTS):
            # Générer les données
            statement_data = generate_statement_data()
            
            # Créer le nom du fichier image
            file_name = f"statement_{i:04d}.png"
            output_path = os.path.join(OUTPUT_DIR, file_name)
            
            # Créer l'image
            create_statement_image(statement_data, output_path)
            
            # Préparer la ligne à écrire dans le CSV
            row_to_write = {
                "file_name": file_name,
                "account_holder": statement_data['account_holder'],
                "bank_name": statement_data['bank_name'],
                "iban": statement_data['iban'],
                "period": statement_data['period'],
                "start_balance": statement_data['start_balance'],
                "end_balance": statement_data['end_balance'],
                # On stocke les transactions en format JSON pour une relecture facile
                "transactions_json": str(statement_data['transactions'])
            }
            writer.writerow(row_to_write)
            
            print(f"  ({i + 1}/{TOTAL_STATEMENTS}) Image {file_name} générée.", end='\r')

    print(f"\n\nTerminé ! {TOTAL_STATEMENTS} images et leur ground_truth.csv sont prêts dans le dossier '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()