# Установка MiniMax Music 3

## Системные требования

- **GPU**: NVIDIA RTX 3090 (24 GB VRAM) или аналогичный
- **CUDA**: 12.4
- **Python**: 3.12.13 (через pyenv)
- **Диск**: ~54 GB для модели

## Пошаговая установка

### 1. Создание виртуального окружения

```bash
cd ~/workspace/projects/learn_minimax-music3

# Создание venv
python3.12 -m venv ~/workspace/venvs/learn_minimax-music3/cuda_12_4/

# Активация
source ~/workspace/venvs/learn_minimax-music3/cuda_12_4/bin/activate
```

### 2. Установка зависимостей

```bash
pip install --upgrade pip setuptools wheel

# Основные зависимости
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

pip install diffusers==0.40.0 transformers==5.16.1 accelerate==1.14.0

# Аудио и анализ
pip install soundfile librosa datasets

# Дополнительные
pip install peft bitsandbytes gradio jupyter matplotlib

# Инструменты разработки
pip install ruff mypy pytest
```

### 3. Скачивание модели

```bash
# Вариант A: через скрипт
python scripts/download_model.py

# Вариант B: через huggingface-cli
huggingface-cli download MiniMaxAI/MiniMax-Music3 \
    --local-dir ~/workspace/models/models_music/MiniMax-Music3
```

### 4. Создание modular_config.json

Модель требует файл `modular_config.json` для работы с `ModularPipeline`.
Создайте его в директории модели:

```bash
cp modular_config.json ~/workspace/models/models_music/MiniMax-Music3/
```

### 5. Проверка установки

```bash
source .venv
python src/lesson_01_model_intro.py
```

Ожидаемый результат:
- Загрузка модели за ~7 секунд
- VRAM allocated: ~22.4 GB
- Все компоненты загружены

## Важные замечания

1. **PYTORCH_CUDA_ALLOC_CONF**: Для нескольких генераций в одной сессии установите:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
   (Уже включено в `src/common.py`)

2. **local_files_only=True**: Все загрузки компонентов используют `local_files_only=True` для исключения сетевых запросов.

3. **trust_remote_code=True**: Требуется для загрузки Qwen3ForCausalLM.

4. **VRAM**: Модель занимает ~22.4 GB на RTX 3090. Для нескольких генераций убедитесь, что `PYTORCH_CUDA_ALLOC_CONF` установлен.

## Структура модели

```
MiniMax-Music3/
├── language_model/          # Global LLM (Qwen3, 8B)
├── transformer/             # Flow Matching (2.4B)
├── condition_encoder/       # Condition Encoder
├── rvq_depth_decoder/       # RVQ Decoder
├── vocoder/                 # Vocoder
├── scheduler/               # FlowMatchEulerDiscreteScheduler
├── tokenizer/               # Qwen2Tokenizer
├── modular_config.json      # Конфигурация pipeline
├── modular_model_index.json # Индекс модели
└── config.json              # Основной конфиг
```