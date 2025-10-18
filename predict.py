# predict.py

import pandas as pd
import sys
import os

# Добавляем src в путь поиска модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predictor import HeartAttackPredictor

def main():
    # Пути
    test_path = 'data/heart_test.csv'
    output_path = 'submission.csv'
    
    # Загрузка
    df_test = pd.read_csv(test_path)
    
    # Предсказание
    predictor = HeartAttackPredictor(
        model_path='models/heart_attack_model.pkl',
        preprocessor_path='models/preprocessor.pkl'
    )
    predictions = predictor.predict(df_test)
    
    # Сохранение
    submission = pd.DataFrame({
        'id': df_test['id'],
        'prediction': predictions.astype(int)
    })
    submission.to_csv(output_path, index=False)
    print(f"✅ Готово! Результат сохранён в {output_path}")

if __name__ == "__main__":
    main()