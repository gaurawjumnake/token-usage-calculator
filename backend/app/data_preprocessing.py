import pandas as pd
import re
import json

fname = "backend/scrapped_data/openrouter.json"

# def preprocess_data(fname,):
df =  pd.read_json(fname)
df['provider'] = df['id'].apply(lambda x: x.split('/')[0])
df['model_id'] = df['id'].apply(lambda x: x.split('/')[1])
df.drop(columns=['id', 'canonical_slug', 'hugging_face_id'], inplace=True)
df['provider'] = df['provider'].apply(lambda x: re.sub(r'[^a-zA-Z0-9-]', '', x))
df['model_id'] = df['model_id'].apply(lambda x: re.sub(r'[^a-zA-Z0-9-]', '', x))
for col in df.columns:
    print(f"Processing column: {col} with dtype: {df[col].dtype}")
    if df[col].dtype == 'object' or df[col].dtype.name == 'str':
        if df[col].isnull().any():
            df[col] = df[col].fillna('', inplace=True).astype(str).str.strip()
        else:
            continue
    else:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0, inplace=True)
        else:
            continue

# model_ids = set(df['model_id'].unique())
catalog = df.set_index("model_id").to_dict(orient="index")
with open("backend/data/model_catalog.json", "w", encoding="utf-8") as f:
    json.dump(
        catalog,
        f,
        indent=2,
        ensure_ascii=False,
        default=str
    )

# with open("backend/processed_data/list_model_ids.txt", "w") as f:
#     for model_id in model_ids:
#         f.write(model_id + "\n")
# print(df.columns)
# print(df.head())
# print(df.describe())
# df.to_excel("backend/processed_data/openrouter_models.xlsx", index=False)
# df.to_json("backend/processed_data/openrouter_models.json", orient='records', indent=4, index=False)

