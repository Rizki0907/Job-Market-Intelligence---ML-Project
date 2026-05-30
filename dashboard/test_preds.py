import pandas as pd
import numpy as np
import os
import joblib

DATA_DIR = 'd:/Data Science/Semester 4/ML/Code/PROJECT UAS/data/processed'
MODELS_DIR = 'd:/Data Science/Semester 4/ML/Code/PROJECT UAS/models'

df = pd.read_parquet(os.path.join(DATA_DIR, 'jobs_clean.parquet'))
df_reg = df.dropna(subset=['salary_mid']).copy()
df_reg = df_reg[df_reg['salary_mid'].between(30_000, 250_000)].reset_index(drop=True)

sbert = np.load(os.path.join(DATA_DIR, 'sbert_regression_embeddings.npy'))
tfidf = np.load(os.path.join(DATA_DIR, 'tfidf_svd_embeddings.npy'))

te = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.joblib'))
model = joblib.load(os.path.join(MODELS_DIR, 'stacking.joblib'))

import re
def infer_seniority(title):
    t = str(title).lower()
    for pattern, level in [
        (r'\b(intern|internship|co.?op)\b', 0),
        (r'\b(entry.?level|junior|jr\.?|graduate|trainee)\b', 1),
        (r'\b(associate|mid.?level|ii)\b', 2),
        (r'\b(senior|sr\.?|staff|iii|iv)\b', 3),
        (r'\b(lead|principal|manager|head|supervisor)\b', 4),
        (r'\b(director|vp|vice|chief|cto|cfo|ceo|president)\b', 5),
    ]:
        if re.search(pattern, t): return level
    return -1

df_reg['seniority_from_title'] = df_reg['title'].apply(infer_seniority)
recovered = (df_reg['seniority_encoded'] == -1) & (df_reg['seniority_from_title'] != -1)
df_reg.loc[recovered, 'seniority_encoded'] = df_reg.loc[recovered, 'seniority_from_title']

still_unknown = df_reg['seniority_encoded'] == -1
ind_median = df_reg[~still_unknown].groupby('industry')['seniority_encoded'].median()
glob_median = df_reg.loc[~still_unknown, 'seniority_encoded'].median()
df_reg['seniority_encoded'] = df_reg.apply(
    lambda r: r['seniority_encoded'] if r['seniority_encoded'] != -1 else ind_median.get(r['industry'], glob_median),
    axis=1
).round().astype(int)

KEYWORDS = ["python", "sql", "aws", "azure", "gcp", "java", "javascript", "typescript", "react", "docker", "kubernetes", "spark", "machine learning", "deep learning", "data", "manager", "director", "lead", "senior", "principal", "architect", "agile", "scrum", "tableau", "excel", "r programming", "tensorflow", "pytorch", "compliance", "equity", "hedge fund", "private equity", "investment banking", "c++", "scala", "golang", "security", "devops", "mlops", "product manager", "data scientist", "data engineer", "software engineer", "quantitative", "physician", "dentist", "attorney", "surgeon", "pharmacist", "portfolio", "actuar", "underwriter", "cloud", "blockchain", "embedded"]
kw_flags = np.array(df_reg['text_raw'].fillna('').str.lower().apply(lambda t: [1 if k in t else 0 for k in KEYWORDS]).tolist())
kw_cols = [f"kw_{k.replace(' ', '_').replace('+', 'plus').replace('-', '_')}" for k in KEYWORDS]
for i, c in enumerate(kw_cols): df_reg[c] = kw_flags[:, i]

df_reg['company_size_enc'] = pd.to_numeric(df_reg['company_size'], errors='coerce')
df_reg['company_size_enc'] = df_reg['company_size_enc'].fillna(df_reg['company_size_enc'].median())
for c in ['location', 'industry']:
    if c in df_reg.columns: df_reg[c] = df_reg[c].fillna('Unknown')

df_reg_enc = df_reg.copy()
df_reg_enc[[c + '_te' for c in ['location', 'industry']]] = te.transform(df_reg[['location', 'industry']])
df_reg_enc = pd.get_dummies(df_reg_enc, columns=['formatted_work_type'], drop_first=True, dtype=int)
work_type_cols = [c for c in df_reg_enc.columns if c.startswith('formatted_work_type_')]

df_reg_enc['interact_sen_loc'] = df_reg_enc['seniority_encoded'] * df_reg_enc['location_te']
df_reg_enc['interact_sen_exp'] = df_reg_enc['seniority_encoded'] * df_reg_enc['years_exp_req']
df_reg_enc['interact_exp_ind'] = df_reg_enc['years_exp_req'] * df_reg_enc['industry_te']

df_reg_enc = pd.get_dummies(df_reg_enc, columns=['seniority_encoded'], drop_first=True, dtype=int)
sen_cols = [c for c in df_reg_enc.columns if c.startswith('seniority_encoded_')]

num_cols = (
    ['years_exp_req', 'education_level', 'is_remote', 'company_size_enc']
    + [c + '_te' for c in ['location', 'industry']]
    + work_type_cols 
    + sen_cols 
    + kw_cols
    + ['interact_sen_loc', 'interact_sen_exp', 'interact_exp_ind']
)

X_num = df_reg_enc[num_cols].fillna(0).values.astype(np.float32)
X_all = np.hstack([sbert, tfidf, X_num])

import warnings
warnings.filterwarnings('ignore')

preds = model.predict(X_all[:10])
print('Num cols count:', len(num_cols))
print('Work type cols:', work_type_cols)
print('Sen cols:', sen_cols)
print('X_num shape:', X_num.shape)
print('Raw predictions:', preds)
print('Expm1 predictions:', np.expm1(preds))
