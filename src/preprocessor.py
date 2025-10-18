import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.medians = {}
        self.modes = {}
        self.leaky_features = ['Unnamed: 0', 'id', 'Blood sugar', 'CK-MB', 'Troponin']
        self.cols_with_nan = [
            'Diabetes', 'Family History', 'Smoking', 'Obesity',
            'Alcohol Consumption', 'Previous Heart Problems',
            'Medication Use', 'Stress Level', 'Physical Activity Days Per Week'
        ]

    def fit(self, df: pd.DataFrame):
        for col in self.cols_with_nan:
            if col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    self.medians[col] = df[col].median()
                else:
                    mode_series = df[col].mode()
                    self.modes[col] = mode_series[0] if not mode_series.empty else None
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols_to_drop = [c for c in self.leaky_features if c in df.columns]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        for col in self.cols_with_nan:
            if col in df.columns:
                if col in self.medians:
                    df[col] = df[col].fillna(self.medians[col])
                elif col in self.modes and self.modes[col] is not None:
                    df[col] = df[col].fillna(self.modes[col])
        
        binary_cols = [
            'Diabetes', 'Family History', 'Smoking', 'Obesity',
            'Alcohol Consumption', 'Previous Heart Problems',
            'Medication Use', 'Stress Level'
        ]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].round().astype('int8')
        
        if 'Diet' in df.columns:
            df['Diet'] = df['Diet'].round().astype('int8')
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].astype('category')
        
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)