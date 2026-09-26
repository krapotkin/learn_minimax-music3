# Руководство по написанию промптов для MiniMax Music 3

## Формат Structured Caption

Рекомендуемый формат описания стиля состоит из трёх секций:

### 1. Global Metadata

```
Genre: [жанр]. BPM: [темп]. Key: [тональность]. Scale: [лад].
Emotional progression: [описание эмоционального развития].
Application Scenarios & Imagery: [контекст/атмосфера].
Sonics & Production Profile: [характеристики продакшна].
```

Параметры:

- **Genre**: pop, rock, metal, jazz, classical, electronic, hip hop, blues, reggae, country, folk, ambient, synthwave, progressive rock, delta blues, jazz fusion, progressive house, acoustic pop
- **BPM**: 40-200 (стандарт: 60-180)
- **Key**: C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B (+ major/minor)
- **Scale**: major, minor, dorian, phrygian, lydian, mixolydian, etc.

### 2. Vocal Details

```
Vocal Gender & Timbre: [пол]. [описание тембра].
Vocal Style: [стиль исполнения].
Harmony/Backing Vocals: [гармонии/бэк-вокал].
Vocal FX: [эффекты].
```

Примеры тембра:

- Female: warm alto, ethereal soprano, breathy, powerful, velvet contralto
- Male: deep baritone, raspy tenor, gritty, smooth, theatrical

Примеры эффектов:

- Plate reverb, slapback delay, vocoder, pitch shifting, distortion, tube preamp

### 3. Arrangement

```
Primary: [основные инструменты].
Secondary: [дополнительные инструменты].
Groove & Foundation: [ритм-секция].
Embellishments, Textures & Spatial FX: [текстуры и эффекты].
```

## Примеры промптов

### Минимальный

```
Genre: pop. BPM: 120. Female vocals, synth and drums.
```

### Средний

```
Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate.
Vocals: soft female lead, close and breathy, light stacked harmonies.
Arrangement: fingerpicked guitar and soft piano; brushed drums in chorus.
```

### Полный (Structured Caption)

```
Global Metadata
Genre: progressive rock. BPM: 100. Key: E minor.
Emotional progression: starts atmospheric, builds to powerful choruses, ends with extended instrumental.

Vocal Details
Vocal Gender & Timbre: Male. Powerful baritone with theatrical delivery.
Vocal Style: Dynamic range from whispered verses to belted choruses.
Harmony/Backing Vocals: Layered harmonies in choruses, call-and-response in bridge.
Vocal FX: Moderate reverb, slight delay on ad-libs.

Arrangement
Primary: Electric guitar with clean-to-overdriven tones, melodic bass guitar.
Secondary: Hammond organ, string section in choruses.
Groove: Dynamic drumming with build-ups and breakdowns.
Textures: Atmospheric pads, guitar feedback swells, orchestral hits.
```

## Теги секций в лирике

Стандартные:

- `[Intro]` -- вступление
- `[Verse]` -- куплет
- `[Pre-Chorus]` -- пре-хорус
- `[Chorus]` -- припев
- `[Post-Chorus]` -- пост-хорус
- `[Bridge]` -- бридж
- `[Instrumental]` -- инструментальная секция
- `[Solo]` -- соло
- `[Outro]` -- концовка
- `[Interlude]` -- интерлюдия

Кастомные (описательные):

- `[bass-quartet-rumbles-in]`
- `[guitar-solo]`
- `[drum-fill]`
- `[key-change]`

Кастомные теги позволяют указать специфические переходы или события.

## Советы

1. **Будьте конкретны.** "acoustic pop" лучше, чем просто "pop"
2. **Указывайте BPM.** Это напрямую влияет на темп
3. **Описывайте вокал.** Пол, тембр и стиль сильно влияют на результат
4. **Используйте теги секций.** Это помогает модели понять структуру
5. **Описывайте развитие.** Модель поддерживает эмоциональную прогрессию
6. **Английский язык.** Промпты на английском дают лучшие результаты