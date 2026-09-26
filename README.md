# MiniMax Music 3 - Образовательный проект

Образовательный проект для изучения модели генерации музыки [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3).

## Архитектура модели

- **Global LLM**: Qwen3-based, 8B параметров, 36 слоёв, hidden=4096
- **Local LLM**: AbabForCausalLM, 0.6B параметров, 36 слоёв, hidden=4096
- **Flow Matching**: 2.4B параметров
- **Flow-VAE**: 123M параметров
- **RVQ Tokenizer**: 8 слоёв (1 semantic x16384 + 7 acoustic x1024)
- **Output**: WAV, 44100 Hz, 16-bit stereo, до 300 секунд

## Быстрый старт

```bash
# Активация окружения
source .venv

# Проверка окружения
python src/lesson_01_model_intro.py

# Генерация музыки
python src/lesson_02_basic_inference.py
```

## Уроки

| # | Файл | Тема |
|---|------|------|
| 01 | `lesson_01_model_intro.py` | Проверка окружения, архитектура, загрузка модели |
| 02 | `lesson_02_basic_inference.py` | Базовая генерация с prompt и lyrics |
| 03 | `lesson_03_lyrics_control.py` | Управление структурой через теги секций |
| 04 | `lesson_04_style_prompts.py` | Детальные описания стиля (Structured Caption) |
| 05 | `lesson_05_duration_control.py` | Контроль длительности генерации |
| 06 | `lesson_06_multi_genre.py` | Генерация в разных жанрах |
| 07 | `lesson_07_vram_optimization.py` | VRAM оптимизация, анализ памяти |
| 08 | `lesson_08_reproducibility.py` | Воспроизводимость, детерминированность |
| 09 | `lesson_09_quality_evaluation.py` | Оценка качества, метрики |

## Технические характеристики

- **VRAM**: ~22.4 GB (RTX 3090 24 GB)
- **Загрузка модели**: ~7 секунд
- **Генерация 10 сек**: ~23 секунды
- **Фреймрейт**: 25 фреймов/сек
- **Частота дискретизации**: 44100 Hz

## Структура

```
learn_minimax-music3/
├── src/
│   ├── common.py              # Общий модуль загрузки и генерации
│   ├── lesson_01_model_intro.py
│   ├── lesson_02_basic_inference.py
│   ├── lesson_03_lyrics_control.py
│   ├── lesson_04_style_prompts.py
│   ├── lesson_05_duration_control.py
│   ├── lesson_06_multi_genre.py
│   ├── lesson_07_vram_optimization.py
│   ├── lesson_08_reproducibility.py
│   └── lesson_09_quality_evaluation.py
├── scripts/
│   └── download_model.py      # Скачивание модели с HuggingFace
├── docs/
│   ├── architecture.md        # Архитектура модели
│   └── style_guide.md         # Руководство по промптам
├── requirements.txt
├── pyproject.toml
├── INSTALL.md
└── AGENTS.md
```

## Выходные данные

Аудиофайлы сохраняются в `~/workspace/tmp/learn_minimax-music3/lesson_XX/`.