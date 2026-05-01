# About
Минимальный проект для обучения генератора подписей к изображениям на связке `ResNet50 encoder + LSTM decoder + attention`.

Что реализовано:
- обучение в `PyTorch`;
- метрики `BLEU` и `METEOR`;
- чекпоинты;
- конфигурация через `yaml`;
- опциональный `ClearML`;
- backbone resnet50 (опционально).

## Installation
```sh
git clone https://github.com/haritonn/caption_gen
cd caption_gen
uv sync
```

## Launch
```sh
chmod +x ./entrypoint.sh
./entrypoint.sh
```
