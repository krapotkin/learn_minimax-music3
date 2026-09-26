"""
Lesson 12b: Песня про Аню Утробину — перегенерация

Проблема: неверное ударение (модель поёт УтробИна вместо УтрОбина).
Подходы:
1. Другой seed
2. Намёк на ударение в промпте
3. Имя в начале строки (падение на сильную долю)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import setup_logging, load_pipeline, generate_music, save_audio, get_vram_usage

logger = setup_logging(__name__, "lesson_12b")

LYRICS_V1 = """[Verse 1]
В нашем городе живёт одна девица
Смех её звенит, как колокольчик
Утробина Аня — имя необычное
Но характер — просто восхитительный

[Chorus]
Аня Утробина, Аня Утробина
С ней веселей любая минута
Аня Утробина, Аня Утробина
Где она — там музыка и смех

[Verse 2]
Она поёт, танцует, вдохновляет
Вся её жизнь — как праздник
Аня Утробина — это не просто имя
Это настроение, это сила

[Chorus]
Аня Утробина, Аня Утробина
С ней веселей любая минута
Аня Утробина, Аня Утробина
Где она — там музыка и смех

[Bridge]
Пусть звучит её имя во все концы
Аня Утробина — звезда нашей тьмы
С ней любой день становится светлым
И душа поёт, как в первый раз

[Chorus]
Аня Утробина, Аня Утробина
С ней веселей любая минута
Аня Утробина, Аня Утробина
Где она — там музыка и смех
"""

LYRICS_V2 = """[Verse 1]
Утробина Аня — наша городская звезда
Её смех слышно, даже если далеко
Она входит — и сразу веселей
Каждый день с ней — маленький праздник

[Chorus]
Утробина Аня, Утробина Аня
С ней любая минута светлая
Утробина Аня, Утробина Аня
Где она — там музыка и смех

[Verse 2]
Утробина Аня поёт и танцует
Вдохновляет всех вокруг
Её имя — это настроение
Это сила, это свет

[Chorus]
Утробина Аня, Утробина Аня
С ней любая минута светлая
Утробина Аня, Утробина Аня
Где она — там музыка и смех

[Bridge]
Пусть звучит её имя во все концы
Утробина Аня — звезда наша
С ней любой день становится светлым
И душа поёт, как в первый раз

[Chorus]
Утробина Аня, Утробина Аня
С ней любая минута светлая
Утробина Аня, Утробина Аня
Где она — там музыка и смех
"""

def main():
    logger.info("="*60)
    logger.info("Урок 12b: Перегенерация — правильное ударение")
    logger.info("="*60)

    logger.info("Загрузка модели...")
    pipe = load_pipeline()
    logger.info(f"VRAM после загрузки: {get_vram_usage()} MiB")

    output_dir = os.path.expanduser("~/workspace/tmp/learn_minimax-music3/lesson_12b")
    os.makedirs(output_dir, exist_ok=True)

    # Вариант 1: оригинальный текст, другой seed, намёк на ударение
    logger.info("\n[1/2] Вариант 1: seed=100, намёк на ударение...")
    t0 = time.time()
    result = generate_music(
        pipe,
        prompt="Russian rock song, female vocals, electric guitar, powerful, energetic, 130 BPM. IMPORTANT: the surname Utrobina has stress on the second syllable: Ut-RO-bi-na (УтрОбина)",
        lyrics=LYRICS_V1,
        duration=90.0,
        seed=100
    )
    wav_path = save_audio(result, os.path.join(output_dir, "anya_v1.wav"))
    logger.info(f"  Сохранено: {wav_path} ({time.time()-t0:.1f}с)")

    # Очистка памяти
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # Вариант 2: перестроенный текст (Утробина Аня), другой seed
    logger.info("\n[2/2] Вариант 2: 'Утробина Аня', seed=200...")
    t0 = time.time()
    result = generate_music(
        pipe,
        prompt="Russian rock song, female vocals, electric guitar, powerful, energetic, 130 BPM. IMPORTANT: the surname Utrobina has stress on the second syllable: Ut-RO-bi-na (УтрОбина)",
        lyrics=LYRICS_V2,
        duration=90.0,
        seed=200
    )
    wav_path = save_audio(result, os.path.join(output_dir, "anya_v2.wav"))
    logger.info(f"  Сохранено: {wav_path} ({time.time()-t0:.1f}с)")

    # Конвертация в OGG
    logger.info("\nКонвертация в OGG...")
    import subprocess
    for wav in ["anya_v1.wav", "anya_v2.wav"]:
        ogg = wav.replace(".wav", ".ogg")
        subprocess.run(
            ["ffmpeg", "-i", os.path.join(output_dir, wav),
             "-codec:a", "libvorbis", "-b:a", "128k",
             os.path.join(output_dir, ogg)],
            capture_output=True
        )
        os.remove(os.path.join(output_dir, wav))

    logger.info("\n" + "="*60)
    logger.info("Урок 12b завершён!")
    logger.info(f"Результаты: {output_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()