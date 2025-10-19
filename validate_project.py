# validate_project.py

import os
import pandas as pd
import sys

def check_file_exists(path, description):
    if os.path.exists(path):
        print(f"✅ {description}: найден")
        return True
    else:
        print(f"❌ {description}: отсутствует")
        return False

def validate_submission():
    path = "submission.csv"
    if not check_file_exists(path, "Файл предсказаний (submission.csv)"):
        return False

    try:
        df = pd.read_csv(path)
        required_cols = {"id", "prediction"}
        if set(df.columns) != required_cols:
            print(f"❌ submission.csv: ожидаемые колонки {required_cols}, получены {set(df.columns)}")
            return False
        if len(df) != 966:
            print(f"❌ submission.csv: ожидалось 966 строк, получено {len(df)}")
            return False
        if not df["prediction"].isin([0, 1]).all():
            print("❌ submission.csv: колонка 'prediction' должна содержать только 0 и 1")
            return False
        print("✅ submission.csv: корректный формат")
        return True
    except Exception as e:
        print(f"❌ Ошибка при чтении submission.csv: {e}")
        return False

def validate_structure():
    required_dirs = ["notebooks", "src", "models"]
    required_files = {
        "notebooks/Heart_Project.ipynb": "Jupyter Notebook с исследованием",
        "src/__init__.py": "Инициализация модуля src",
        "src/preprocessor.py": "Класс DataPreprocessor",
        "src/predictor.py": "Класс HeartAttackPredictor",
        "app.py": "FastAPI-сервис",
        "predict.py": "CLI-скрипт",
        "README.md": "Документация",
        "base.txt": "Зависимости"
    }

    all_ok = True

    for d in required_dirs:
        if not os.path.isdir(d):
            print(f"❌ Папка '{d}' отсутствует")
            all_ok = False
        else:
            print(f"✅ Папка '{d}' найдена")

    for f, desc in required_files.items():
        if not check_file_exists(f, desc):
            all_ok = False

    # Проверка: нет ли кода приложения в ноутбуке (косвенно)
    notebook_path = "notebooks/Heart_Project.ipynb"
    if os.path.exists(notebook_path):
        with open(notebook_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "class DataPreprocessor" in content or "class HeartAttackPredictor" in content:
                print("⚠️  В ноутбуке обнаружен код классов — он должен быть в src/")
                all_ok = False
            else:
                print("✅ Код классов отсутствует в ноутбуке")

    return all_ok

def main():
    print("🔍 Проверка структуры проекта по ТЗ...\n")

    ok1 = validate_structure()
    ok2 = validate_submission()

    print("\n" + "="*50)
    if ok1 and ok2:
        print("🎉 Проект прошёл все проверки! Готов к сдаче.")
        sys.exit(0)
    else:
        print("🚨 Найдены ошибки. Исправьте их перед сдачей.")
        sys.exit(1)

if __name__ == "__main__":
    main()