import joblib
import pandas as pd
import numpy as np

DATA_DIR = 'd:/Data Science/Semester 4/ML/Code/PROJECT UAS/data/processed'
MODELS_DIR = 'd:/Data Science/Semester 4/ML/Code/PROJECT UAS/models'

df = pd.read_parquet(DATA_DIR + '/jobs_clean.parquet')
df_reg = df.dropna(subset=['salary_mid']).copy()
df_reg = df_reg[df_reg['salary_mid'].between(30_000, 250_000)].reset_index(drop=True)
sbert = np.load(DATA_DIR + '/sbert_regression_embeddings.npy')
tfidf = np.load(DATA_DIR + '/tfidf_svd_embeddings.npy')
te = joblib.load(MODELS_DIR + '/target_encoder.joblib')
model = joblib.load(MODELS_DIR + '/stacking.joblib')

# Just generate the first row features manually for 604 features
# ... actually we can just copy app.py's load_simulator_data exactly to see if it works.

import app
model, te, sample_df, sample_X = app.load_simulator_data()
preds = model.predict(sample_X[:5])
print("Sample preds:", np.expm1(preds))
print("Actuals:", sample_df['salary_mid'].head().values)
