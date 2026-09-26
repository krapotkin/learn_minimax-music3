# Архитектура MiniMax Music 3

## Обзор

MiniMax Music 3 -- модель генерации музыки, использующая гибридную архитектуру из двух языковых моделей и системы непрерывного синтеза на основе Flow Matching.

```
Text Input (lyrics + style description)
         |
         v
  +---------------------+
  |  Condition Encoder  |
  +---------------------+
         |
         v
  +---------------------+     +------------------+
  |   Global LLM (8B)   |---->|                  |
  |  (Qwen3-based)      |     |  Hidden-state    |
  |  First RVQ codebook |     |  Fusion          |
  +---------------------+     |                  |
         |                    +--------+---------+
         v                             |
  +---------------------+              |
  |   Local LLM (0.6B)  |--------------+
  |  (AbabForCausalLM)  |
  |  Remaining RVQ      |
  |  codebooks          |
  +---------------------+
         |
         v
  +---------------------+
  |  Flow Matching (2.4B)|
  |  Continuous latent   |
  +---------------------+
         |
         v
  +---------------------+
  |  Flow-VAE (123M)    |
  |  Latent decoder     |
  +---------------------+
         |
         v
  32 kHz stereo WAV
```

## Компоненты

### 1. Music Tokenizer (RVQ)

Residual Vector Quantization с 8 слоями:

- **Semantic codebook** (layer 0): 16,384 entries -- основная музыкальная семантика
- **Acoustic codebooks** (layers 1-7): по 1,024 entries -- остаточные акустические детали

Общий токенов на фрейм: 8 codebooks.
Фреймов в секунду: 25.

### 2. Condition Encoder

Кодирует текстовый ввод (лирику + описание стиля) в условие для языковых моделей. Использует diffusion-based подход.

### 3. Global LLM (8B)

- **Архитектура**: Qwen3ForCausalLM
- **Параметры**: ~8B
- **Слои**: 36
- **Hidden size**: 4096
- **Attention heads**: 32 (GQA, 8 KV heads)
- **Vocab size**: 200,000
- **Max position embeddings**: 10,240
- **Роль**: Предсказание первого RVQ codebook фрейм за фреймом. Моделирует долгосрочную музыкальную структуру и семантику песни.
- **Инициализация**: Qwen3-8B с адаптацией embedding и output layers к музыкальным токенам.

### 4. Local LLM (0.6B)

- **Архитектура**: AbabForCausalLM
- **Параметры**: ~0.6B
- **Слои**: 36
- **Hidden size**: 4096
- **Attention heads**: 32 (GQA, 8 KV heads)
- **Vocab size**: 200,000
- **Max position embeddings**: 10,240
- **Роль**: Предсказание оставшихся 7 RVQ codebooks внутри каждого фрейма. Восстанавливает детальные акустические характеристики.

### 5. Hidden-state Fusion

Вместо использования только дискретных RVQ токенов, система объединяет непрерывные скрытые состояния из обоих LLM. Это сохраняет более богатую акустическую информацию для:

- Вокальной артикуляции
- Инструментальной текстуры
- Временной непрерывности

### 6. Flow Matching (2.4B)

- **Параметры**: ~2.4B
- **Роль**: Преобразование объединённых скрытых состояний в латентное пространство Flow-VAE
- **Подход**: Flow Matching -- альтернатива диффузионным моделям, обеспечивающая более быстрый и стабильный синтез

### 7. Flow-VAE (123M)

- **Параметры**: ~123M
- **Архитектура**: Адаптирована от MiniMax Speech, переобучена для музыки
- **Роль**: Декодирование латентного представления в аудиоволну (32 kHz, 16-bit stereo)

## Входные данные

### Лирика (Lyrics)

Текст песни с опциональными тегами секций:

```
[Intro]
[Verse]
...текст...
[Pre-Chorus]
...текст...
[Chorus]
...текст...
[Bridge]
...текст...
[Instrumental]
[Solo]
[Outro]
```

### Описание стиля (Music Description / Structured Caption)

Рекомендуемый формат с тремя секциями:

1. **Global Metadata**: genre, subgenre, BPM, key, scale, emotional progression, production profile
2. **Vocal Details**: gender, timbre, performance style, harmony, backing vocals, vocal effects
3. **Arrangement**: instruments, section evolution, groove, bass, percussion, textures, spatial effects

## Выходные данные

- **Формат**: WAV
- **Частота дискретизации**: 32,000 Hz
- **Разрешение**: 16-bit
- **Каналы**: 2 (стерео)
- **Максимальная длительность**: 5 минут (300 секунд)
- **Фреймы в секунду**: 25

## VRAM требования

| Режим | VRAM | Скорость |
|-------|------|----------|
| Full precision (BF16) | 24+ GB | Быстрая |
| CPU offloading | ~22 GB | Средняя |
| Layer streaming | 8+ GB | Медленная |

## Размер модели

| Компонент | Файлов | Примерный размер |
|-----------|--------|------------------|
| language_model (Global LLM) | 4 | ~16 GB |
| qwen_7B (Local LLM) | 48 | ~1.2 GB |
| transformer (Flow Matching) | 2 | ~4.8 GB |
| condition_encoder | 1 | ~0.2 GB |
| rvq_depth_decoder | 1 | ~0.1 GB |
| vocoder (Flow-VAE) | 1 | ~0.5 GB |
| tokenizer | 5 | ~500 MB |
| Прочее | - | ~100 MB |
| **Итого** | **~63** | **~50 GB** |

## Ссылки

- HuggingFace: https://huggingface.co/MiniMaxAI/MiniMax-Music3
- GitHub: https://github.com/MiniMax-AI/MiniMax-Music3
- Demo: https://minimax-ai.github.io/music3-demo/
- SGLang-Omni: https://github.com/sgl-project/sglang-omni