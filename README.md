# Генератор подписей к изображениям

Проект обучает модель, которая генерирует англоязычную текстовую подпись для изображения. Используется датасет Flickr8k и архитектура `ResNet50 encoder + attention + LSTM decoder`, реализованная на PyTorch.

## Возможности

- извлечение пространственных признаков изображения с помощью ResNet50;
- soft-attention по сетке признаков `7 × 7` при генерации каждого слова;
- LSTM-декодер с эмбеддингами слов, layer normalization и dropout;
- teacher forcing с подстановкой предсказанных токенов (`sampling_prob: 0.35`);
- label smoothing, L2-регуляризация внимания и gradient clipping;
- детерминированное разбиение изображений на train/validation/test;
- оценка качества генерации по BLEU и METEOR;
- early stopping, сохранение чекпоинтов и опциональное логирование в ClearML.

## Результат последнего обучения

Запуск был остановлен early stopping'ом на 34-й эпохе из запланированных 100. Лучший чекпоинт соответствует 14-й эпохе. [Ссылка на best checkpoint](https://drive.google.com/file/d/1vfhpcsZG2p6B1jFaP-NVQgsAdz3jjw55/view?usp=sharing), должна быть в `checkpoints/`.

| Показатель | Значение |
| --- | ---: |
| Лучшая validation loss | 3.8797 |
| Эпоха лучшего чекпоинта | 14 |
| Train loss | 2.9984 |
| Validation loss | 3.9192 |
| BLEU | 0.1655 |
| METEOR | 0.3774 |
| Vocab size | 4 697 |

Финальные метрики рассчитаны на validation-части разбиения. BLEU считается как corpus BLEU со сглаживанием, а METEOR усредняется по изображениям с несколькими эталонными подписями. Поэтому эти значения стоит сравнивать только с запусками, использующими те же разбиение, предобработку и параметры оценки.

Примеры генерации на финальной эпохе:

```text
Prediction: a dog dog is jumping over a log
Reference:  a black dog leaps over a log

Prediction: a person in a blue jacket is in the snow
Reference:  a man uses ice picks and <UNK> to <UNK> ice

Prediction: a boy boy in shorts is holding a <UNK>
Reference:  a boy in his blue swim shorts at the beach
```
Модель улавливает основные объекты и простые сцены, но заметны повторы слов и `<UNK>` для редких слов. Это ожидаемое ограничение текущего словаря и greedy-декодирования.

## Архитектура

```text
Изображение 224×224
        |
        |
ResNet50 без классификационной головы (замороженный encoder)
        |  7×7×2048 признаков
        |
Attention + LSTM decoder
        |
        |
Последовательность токенов: <START> ... <END>
```

По умолчанию encoder использует предобученные веса `ResNet50_Weights.IMAGENET1K_V1`; его параметры заморожены. Декодер использует размер эмбеддинга 256, скрытое состояние LSTM 512 и attention размерности 512.

## Требования

- Python 3.10 или новее;
- [uv](https://docs.astral.sh/uv/);
- NVIDIA GPU с CUDA - рекомендуется. Если CUDA недоступна, загрузчик конфигурации автоматически переключит устройство на CPU.
## Установка

```sh
git clone https://github.com/haritonn/caption_gen
cd caption_gen
uv sync
```
## Данные

Проект ожидает Flickr8k в следующем виде:

```text
data/
|- captions.txt
|- Images/
    |- 1000268201_693b08cb0e.jpg
    |- ...
```

Файл `captions.txt` должен содержать столбцы `image` и `caption`.

Чтобы скачать датасет через KaggleHub и сразу запустить обучение, передайте `1` скрипту запуска:

```sh
chmod +x ./entrypoint.sh
./entrypoint.sh 1
```

Если данные уже размещены в `data/`, достаточно запустить:

```sh
./entrypoint.sh
```

Эквивалентный прямой запуск:

```sh
uv run python -u train.py
```

## Настройка запуска

Основные параметры находятся в [`config.yaml`](config.yaml).

| Группа | Ключи | Текущие значения |
| --- | --- | --- |
| Данные | `dataset.train_size`, `dataset.val_size` | `0.8`, `0.1` |
| Изображения | `dataset.image_size` | `224` |
| Словарь | `dataset.max_caption_length`, `dataset.min_word_freq` | `50`, `2` |
| Модель | `model.decoder.embedding_dim`, `model.decoder.hidden_dim` | `256`, `512` |
| Оптимизация | `training.batch_size`, `training.learning_rate` | `128`, `0.0002` |
| Scheduler | `step_size`, `gamma` | `25`, `0.5` |
| Регуляризация | `label_smoothing`, `attention_regularization` | `0.1`, `0.001` |
| Hardware | `hardware.device`, `hardware.num_workers` | `cuda`, `16` |

После 25-й эпохи StepLR уменьшает learning rate вдвое: в показанном запуске к 34-й эпохе он составлял `0.000100`.

Для запуска на CPU можно явно установить в `config.yaml`:

```yaml
hardware:
  device: cpu
  num_workers: 4
  pin_memory: false
```

## Чекпоинты и воспроизводимость

Каждый чекпоинт содержит веса модели и оптимизатора, номер эпохи, лучшую validation loss, словарь `word2idx`/`idx2word` и актуальную конфигурацию.

В конфигурации включены `seed: 42` и детерминированный режим cuDNN.

## Опциональный ClearML

Чтобы логировать параметры, архитектуру и метрики в ClearML, измените:

```yaml
hardware:
  experiment_tracking:
    enabled: true
    project_name: image_caption_generator
    experiment_name: my_experiment
```

Без этой настройки проект работает локально и не требует ClearML-сервера.

## Структура проекта

```text
train.py             цикл обучения, валидация и метрики
model/model.py       ResNet50 encoder, attention и LSTM decoder
dataset/dataset.py   Flickr8k dataset, словарь и разбиение
utils/               обработка изображений и подписей
config.yaml          параметры эксперимента
install_data.py      загрузка Flickr8k через KaggleHub
entrypoint.sh        загрузка данных (опционально) и запуск обучения
checkpoints/         сохранённые веса, включая best_model.pth
```
