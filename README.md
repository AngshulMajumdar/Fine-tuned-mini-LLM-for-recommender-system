# MovieLens Fine-Tuned Recommendation API

FastAPI service for binary recommendation using a fine-tuned causal LM.

## What it does
Given:
- `user_id`
- `item_id`

it returns:
- binary prediction `0/1`
- score for class `1`

## Important
This repo does not include model weights.

Default model path in the generated code:
`/content/drive/MyDrive/movielens_100k_ft_distilgpt2_steps300_clean/model/final_model`

Default prepared-data path in the generated code:
`/content/drive/MyDrive/movielens_100k_ft_distilgpt2_steps300_clean/prepared`

On another machine, override them with environment variables:
- `MOVIELENS_MODEL_DIR`
- `MOVIELENS_PREPARED_DIR`

## Run

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
