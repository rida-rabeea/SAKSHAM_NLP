import os
import pandas as pd
import zipfile
import requests
from transformers import BartTokenizer, BartForSequenceClassification
from pymongo import MongoClient
from dotenv import load_dotenv

def load_model_and_resources():
    load_dotenv()

    model_dir = "final_model"  # Should already exist on Render

    model = BartForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    df = pd.read_csv("intent_mappings.csv", encoding="ISO-8859-1")

    mongo_client = MongoClient(os.getenv("MONGO_URI"))
    collection = mongo_client["ISMDATA"]["ISMDATA"]

    return model, tokenizer, df, collection


 
