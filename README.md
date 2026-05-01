# About
Минимальный проект для обучения генератора подписей к изображениям на связке `ResNet50 encoder + LSTM decoder + attention`.

Что осталось:
- обучение в `PyTorch`;
- подготовка датасета и словаря;
- split по изображениям без data leakage;
- метрики `BLEU` и `METEOR`;
- чекпоинты;
- конфигурация через `yaml`;
- аккуратный опциональный `ClearML`.

## Installation
```sh
git clone https://github.com/haritonn/caption_gen
cd caption_gen
uv sync
```

## Launch
```sh
uv run train.py
```

## Scope
В репозитории оставлен только рабочий training pipeline. Тесты, inference-заглушка, frontend-плейсхолдеры, лишние зависимости и неиспользуемые части удалены. Папка `experiments/` сохранена.
