from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import load_model_and_resources
from utils import process_natural_query

app = FastAPI()

model, tokenizer, df, collection = load_model_and_resources()

class Query(BaseModel):
    text: str

@app.post("/predict")
async def predict(query: Query):
    try:
        result = process_natural_query(model, tokenizer, df, collection, query.text)
        return result
    except Exception as e:
        return {"error": str(e)}
