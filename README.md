```markdown
# Прогноз риска сердечного приступа

Модель машинного обучения для оценки риска сердечного приступа на основе анамнеза пациента.

## 📦 Установка зависимостей

Убедитесь, что у вас установлен Python 3.8+. Затем выполните:

```bash
pip install -r requirements.txt
```

Файл `base.txt` содержит все необходимые зависимости.

## 📂 Структура проекта

```
heart-attack-predictor/
├── notebooks/               # Jupyter Notebook с исследованием и обучением
│   └── Heart_Project.ipynb
├── src/                     # Production-код (ООП)
│   ├── __init__.py
│   ├── preprocessor.py      # Класс DataPreprocessor
│   └── predictor.py         # Класс HeartAttackPredictor
├── models/                  # Сохранённые модель и препроцессор (создаются при обучении)
├── data/                    # Исходные данные (не включены в репозиторий)
├── app.py                   # FastAPI-сервис
├── predict.py               # CLI-скрипт для генерации submission.csv
├── submission.csv           # Финальное предсказание (в требуемом формате)
├── base.txt         # Зависимости
└── README.md                # Документация
```

> ⚠️ **Важно**:  
> - В `notebooks/Heart_Project.ipynb` содержится **только исследование**, обучение и выводы.  
> - **Весь production-код вынесен в `src/`** и импортируется как модуль.  
> - Признаки `Blood sugar`, `CK-MB`, `Troponin` удалены как **косвенные утечки** (измеряются после приступа).

## 🗂️ Почему в репозитории нет файлов в `data/` и `models/`?

- **Файлы в `data/`** (`heart_train.csv`, `heart_test.csv`)  не включены в репозиторий, чтобы избежать возможного дублирования и соблюсти условия распространения данных.
- **Файлы в `models/`** (`heart_attack_model.pkl`, `preprocessor.pkl`) **генерируются автоматически** при запуске ноутбука `notebooks/Heart_Project.ipynb`.

### Как воссоздать файлы:

1. Поместите `heart_train.csv` и `heart_test.csv` в папку `data/`.
2. Запустите ноутбук `notebooks/Heart_Project.ipynb` **до конца** — он создаст:
   - `models/heart_attack_model.pkl`
   - `models/preprocessor.pkl`
   - `submission.csv`

Или используйте CLI-скрипт (см. ниже).

## ▶️ Генерация предсказаний (CLI)

Для генерации файла `submission.csv` выполните:

```bash
python predict.py
```

> 💡 Перед запуском убедитесь, что:
> - папка `data/` содержит `heart_test.csv`,
> - папка `models/` содержит обученные файлы (или запустите обучение).

Результат: файл `submission.csv` с двумя колонками:
- `id` — идентификатор пациента,
- `prediction` — предсказание (0 = низкий риск, 1 = высокий риск).

## 🌐 Запуск FastAPI-сервиса

Запустите сервер:

```bash
uvicorn app:app --reload
```

### Документация API
Откройте в браузере:  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Как отправить запрос
1. Нажмите **POST /predict → Try it out**.
2. Выберите файл `heart_test.csv`.
3. Нажмите **Execute**.

Сервис вернёт JSON-массив вида:
```json
[
  {"id": 7746, "prediction": 0},
  {"id": 4202, "prediction": 0},
  ...
]
```

## 📚 Описание классов

### `DataPreprocessor` (`src/preprocessor.py`)
Обрабатывает сырые данные:
- Удаляет утечки (`Blood sugar`, `CK-MB`, `Troponin`, `id`, `Unnamed: 0`),
- Заполняет пропуски медианой/модой,
- Преобразует типы:
  - Бинарные признаки → `int8`,
  - `Diet` → `int8`,
  - `Gender` → `category`.

### `HeartAttackPredictor` (`src/predictor.py`)
Инкапсулирует модель и препроцессор:
- Загружает сохранённые объекты,
- Принимает `DataFrame`,
- Возвращает предсказания как `pd.Series`.

## 📝 Примечания
- Метрика качества: **F1-score** (обосновано дисбалансом классов: ~65% низкого риска, ~35% высокого).
- Модель: **CatBoostClassifier** с кросс-валидацией (`StratifiedKFold`, 5 фолдов).
- Все признаки нормализованы; данные, вероятно, синтетические.

