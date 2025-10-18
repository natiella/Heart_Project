# src/predictor.py

import pandas as pd
from catboost import CatBoostClassifier, Pool
from .preprocessor import DataPreprocessor
import joblib

class HeartAttackPredictor:
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.cat_features = ['Gender']

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = self.preprocessor.transform(df)
        pool = Pool(X, cat_features=self.cat_features)
        preds = self.model.predict(pool)
        return pd.Series(preds, index=df.index, name='prediction')