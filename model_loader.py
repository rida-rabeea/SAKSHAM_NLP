import os
import pandas as pd
from transformers import BartTokenizer, BartForSequenceClassification
from pymongo import MongoClient
from dotenv import load_dotenv
import zipfile
import requests

def download_and_unzip_from_drive(file_id, destination_folder):
    url = f"https://drive.google.com/uc?export=download&id=1V-sixHIF2uxjPMyVdL6YuN3Me-6Lm2hi"
    zip_path = "model.zip"

    # Download ZIP
    response = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Unzip it
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    os.remove(zip_path)  # Clean up

def load_model_and_resources():
    load_dotenv()
    
    model = BartForSequenceClassification.from_pretrained("./final_model")
    tokenizer = BartTokenizer.from_pretrained("./final_model")
    df = pd.read_csv("intent_mappings.csv",encoding="ISO-8859-1")
    
    mongo_client = MongoClient(os.getenv("MONGO_URI"))
    collection = mongo_client["ISMDATA"]["ISMDATA"]
    
    return model, tokenizer, df, collection
