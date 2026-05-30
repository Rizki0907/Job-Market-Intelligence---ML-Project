import pandas as pd
import numpy as np

DATA_DIR = 'd:/Data Science/Semester 4/ML/Code/PROJECT UAS/data/processed'
df = pd.read_parquet(DATA_DIR + '/jobs_clean.parquet')
df_reg = df.dropna(subset=['salary_mid']).copy()
df_reg = df_reg[df_reg['salary_mid'].between(30_000, 250_000)].reset_index(drop=True)

df_reg_enc = df_reg.copy()
df_reg_enc = pd.get_dummies(df_reg_enc, columns=['formatted_work_type'], drop_first=True, dtype=int)
work_type_cols = [c for c in df_reg_enc.columns if c.startswith('formatted_work_type_')]
print("Work type cols:", work_type_cols)
print("Length:", len(work_type_cols))

# Are there any other dummy columns?
# The notebook used:
# te_cols = [c for c in ["location", "industry"] if c in df_reg.columns]
# te = ce.TargetEncoder(cols=te_cols, smoothing=20)
# This is exactly 2 columns.
