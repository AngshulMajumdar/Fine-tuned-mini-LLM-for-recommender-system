
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_DIR = r"/content/drive/MyDrive/movielens_100k_ft_distilgpt2_steps300_clean/model/final_model"
DEFAULT_PREPARED_DIR = r"/content/drive/MyDrive/movielens_100k_ft_distilgpt2_steps300_clean/prepared"

MODEL_DIR = Path(os.environ.get("MOVIELENS_MODEL_DIR", DEFAULT_MODEL_DIR))
PREPARED_DIR = Path(os.environ.get("MOVIELENS_PREPARED_DIR", DEFAULT_PREPARED_DIR))

app = FastAPI(title="MovieLens Fine-Tuned Recommendation API")

class PredictRequest(BaseModel):
    user_id: int
    item_id: int

class PredictBatchRequest(BaseModel):
    pairs: list[PredictRequest]

def clean_list_field(x):
    if pd.isna(x):
        return []
    x = str(x).strip()
    if not x:
        return []
    return [t.strip() for t in x.split("||") if t.strip()]

def build_prompt(row):
    liked = clean_list_field(row.get("user_liked_movie_titles", ""))
    disliked = clean_list_field(row.get("user_disliked_movie_titles", ""))
    genres = str(row.get("movie_genres", "")).strip()

    liked_text = ", ".join(liked[:6]) if liked else "None"
    disliked_text = ", ".join(disliked[:6]) if disliked else "None"

    return f'''You are a binary movie recommendation classifier. Output only 1 or 0.

User profile:
- age: {row.get("user_age", "unknown")}
- gender: {row.get("user_gender", "unknown")}
- occupation: {row.get("user_occupation", "unknown")}

User history from training data:
- liked movies: {liked_text}
- disliked movies: {disliked_text}

Candidate movie:
- title: {row.get("movie_title", "unknown")}
- genres: {genres if genres else "unknown"}

Answer:'''

@lru_cache(maxsize=1)
def load_assets():
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

    test_csv = PREPARED_DIR / "test_examples.csv"
    if not test_csv.exists():
        raise RuntimeError(f"Prepared test_examples.csv not found: {test_csv}")

    df = pd.read_csv(test_csv)
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    zero_ids = tokenizer(" 0", add_special_tokens=False)["input_ids"]
    one_ids = tokenizer(" 1", add_special_tokens=False)["input_ids"]

    if len(zero_ids) < 1 or len(one_ids) < 1:
        raise RuntimeError("Could not derive token ids for 0 and 1.")

    zero_id = zero_ids[-1]
    one_id = one_ids[-1]

    return df, tokenizer, model, zero_id, one_id

def score_pair(user_id: int, item_id: int):
    df, tokenizer, model, zero_id, one_id = load_assets()

    rows = df[(df["user_id"] == user_id) & (df["item_id"] == item_id)]
    if len(rows) == 0:
        raise HTTPException(status_code=404, detail="Pair not found in prepared test set.")

    row = rows.iloc[0].to_dict()
    prompt = build_prompt(row)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1]
        pair_logits = torch.stack([logits[zero_id], logits[one_id]])
        probs = torch.softmax(pair_logits, dim=0)

    score_1 = float(probs[1].item())
    pred = int(score_1 >= 0.5)

    return {
        "user_id": user_id,
        "item_id": item_id,
        "score_1": score_1,
        "prediction": pred,
        "label_if_present": int(row["label"]) if "label" in row else None,
        "movie_title": row.get("movie_title", "")
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": str(MODEL_DIR),
        "prepared_dir": str(PREPARED_DIR),
        "cuda": torch.cuda.is_available()
    }

@app.get("/meta")
def meta():
    df, tokenizer, model, zero_id, one_id = load_assets()
    return {
        "num_rows_in_test_examples": int(len(df)),
        "model_dir": str(MODEL_DIR),
        "prepared_dir": str(PREPARED_DIR),
        "zero_token_id": int(zero_id),
        "one_token_id": int(one_id),
        "cuda": torch.cuda.is_available()
    }

@app.post("/predict")
def predict(req: PredictRequest):
    return score_pair(req.user_id, req.item_id)

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    return {
        "results": [score_pair(x.user_id, x.item_id) for x in req.pairs]
    }
