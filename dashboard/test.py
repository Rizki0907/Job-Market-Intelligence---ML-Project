import joblib
import pandas as pd
import numpy as np
import os

MODELS_DIR = 'd:/Data Science/Semester 4/ML/Code/PROJECT UAS/models'

te = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.joblib'))
print("Target encoder cols:", te.cols)

model = joblib.load(os.path.join(MODELS_DIR, 'stacking.joblib'))
try:
    print("Stacking features expected:", model.n_features_in_)
    if hasattr(model, 'feature_names_in_'):
        print("Stacking feature names:", model.feature_names_in_)
except Exception as e:
    pass

try:
    xgb_pipe = model.estimators_[0][1]
    if hasattr(xgb_pipe, 'feature_names_in_'):
        print("XGB pipe feature names length:", len(xgb_pipe.feature_names_in_))
        print("Last 20:", xgb_pipe.feature_names_in_[-20:])
except Exception as e:
    print(e)
