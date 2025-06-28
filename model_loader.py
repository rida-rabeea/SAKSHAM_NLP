import os
import pandas as pd
import zipfile
import requests
from transformers import BartTokenizer, BartForSequenceClassification
from pymongo import MongoClient
from dotenv import load_dotenv

def download_and_unzip_from_drive(file_id, destination_folder):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    zip_path = "model.zip"

    # Download the zip file
    response = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    # Delete zip after extraction
    os.remove(zip_path)

def load_model_and_resources():
    load_dotenv()

    model_dir = "final_model"

    # Only download if model folder not present
    if not os.path.exists(model_dir):
        drive_link_or_id = os.getenv("DRIVE_FILE_ID")
        file_id = drive_link_or_id.split("/")[5] if "drive.google.com" in drive_link_or_id else drive_link_or_id
        download_and_unzip_from_drive(file_id, model_dir)

    model = BartForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    df = pd.read_csv("intent_mappings.csv", encoding="ISO-8859-1")

    mongo_client = MongoClient(os.getenv("MONGO_URI"))
    collection = mongo_client["ISMDATA"]["ISMDATA"]

    return model, tokenizer, df, collection
