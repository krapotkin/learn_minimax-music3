"""
Lesson 12: Песня про Аню Утробину

Генерация песни на русском языке с конкретным именем в тексте.
Длительность: 90 секунд (1.5 минуты).
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import setup_logging, load_pipeline, generate_music, save_audio, get_vram_usage

logger = setup_logging(__name__, "lesson_12")

LYRICS = """[Verse 1]
В нашем городе живёт одна девица
Смех её звенит, как колокольчик
Аня Утробина — имя необычное
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

def main():
    logger.info("="*60)
    logger.info("Урок 12: Песня про Аню Утробину")
    logger.info("="*60)

    logger.info("Загрузка модели...")
    pipe = load_pipeline()
    logger.info(f"VRAM после загрузки: {get_vram_usage()} MiB")

    output_dir = os.path.expanduser("~/workspace/tmp/learn_minimax-music3/lesson_12")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("\nГенерация песни, 90 секунд...")
    t0 = time.time()
    result = generate_music(
        pipe,
        prompt="Russian pop song, female vocals, upbeat, cheerful, catchy melody, 120 BPM, modern production",
        lyrics=LYRICS,
        duration=90.0,
        seed=2026
    )
    wav_path = save_audio(result, os.path.join(output_dir, "anya_utrobina.wav"))
    logger.info(f"  Сохранено: {wav_path} ({time.time()-t0:.1f}с)")

    # Конвертация в OGG
    logger.info("\nКонвертация в OGG...")
    import subprocess
    subprocess.run(
        ["ffmpeg", "-i", os.path.join(output_dir, "anya_utrobina.wav"),
         "-codec:a", "libvorbis", "-b:a", "128k",
         os.path.join(output_dir, "anya_utrobina.ogg")],
        capture_output=True
    )
    os.remove(os.path.join(output_dir, "anya_utrobina.wav"))

    logger.info("\n" + "="*60)
    logger.info("Урок 12 завершён!")
    logger.info(f"Результат: {os.path.join(output_dir, 'anya_utrobina.ogg')}")
    logger.info("="*60)

if __name__ == "__main__":
    main()