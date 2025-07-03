import re
import torch
import torch.nn.functional as F
import spacy
import math


nlp = spacy.load("en_core_web_sm")

def extract_filters(sentence):
    filters = {}
    doc = nlp(sentence)

    for ent in doc.ents:
        filters[ent.label_] = ent.text

    ip_matches = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', sentence)
    if ip_matches:
        filters["ip_address"] = ip_matches[0]

    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', sentence)
    if year_match:
        filters["year"] = year_match.group(1)

    return filters


def sanitize_for_json(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return 0
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    return obj

def process_natural_query(model, tokenizer, df, collection, sentence):
    # Tokenize the input
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs.logits

    # Process logits to get top intent predictions
    intent_labels = df["Intent"].unique().tolist()
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:4]
    top_intents = [(intent_labels[i], round(probs[i]*100, 2)) for i in top_indices]
    predicted_intent = top_intents[0][0]

    # Extract filters like IPs, year, etc.
    filters = extract_filters(sentence)

    result = {
        "intent": predicted_intent,
        "top_matches": top_intents,
        "filters": filters,
        "results": None
    }

    # Handle custom intents
    if predicted_intent == "asset_count":
        macs = {doc.get("MAC Address", "").lower() for doc in collection.find({}) if doc.get("MAC Address")}
        result["results"] = {"unique_mac_count": len(macs)}

    elif predicted_intent == "vulnerability_labname":
        lab_counts = collection.aggregate([
            {"$group": {"_id": "$Lab name", "count": {"$sum": 1}}}
        ])
        result["results"] = {doc["_id"]: doc["count"] for doc in lab_counts if doc["_id"]}

    elif predicted_intent == "not_patchable":
        count = collection.count_documents({
            "$or": [{"Solution": None}, {"Solution": ""}, {"Solution": {"$exists": False}}]
        })
        result["results"] = {"not_patchable_count": count}

    elif predicted_intent == "count_vulnerabilities":
        severity_count = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        all_macs = set()
        vuln_macs = set()
        query = {}
        if "year" in filters:
            query["Vuln Publication Date"] = {"$regex": filters["year"]}
        if "ip_address" in filters:
            query["IP Address"] = filters["ip_address"]
        for doc in collection.find(query, {"MAC Address": 1, "Severity": 1}):
            mac, sev = doc.get("MAC Address", "").lower(), doc.get("Severity", "")
            if mac:
                all_macs.add(mac)
                if sev in severity_count:
                    severity_count[sev] += 1
                    vuln_macs.add(mac)
        percent = round((len(vuln_macs) / len(all_macs)) * 100, 2) if all_macs else 0
        result["results"] = {
            "severity_breakdown": severity_count,
            "affected_mac_percentage": percent
        }

    else:
        # Handle DB-based queries with mapped fields
        row = df[df['Intent'] == predicted_intent]
        if row.empty:
            result["error"] = "Intent mapping not found in dataset."
        else:
            entity_fields = [e.strip() for e in row.iloc[0]['Entities'].split(',') if e.strip()]
            projection = {field: 1 for field in entity_fields}
            projection['_id'] = 0

            query = {}
            if "ip_address" in filters:
                query["IP Address"] = filters["ip_address"]
            if "year" in filters:
                date_field = next((e for e in entity_fields if "vuln publication date" in e.lower()), None)
                if date_field:
                    query[date_field] = {"$regex": filters["year"]}

            results = list(collection.find(query, projection))
            result["results"] = results if results else []

    return sanitize_for_json(result)
