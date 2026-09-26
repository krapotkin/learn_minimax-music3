"""
Lesson 10: Русскоязычная генерация

Проверяем способность MiniMax Music 3 генерировать песни на русском языке.
Qwen3 (Global LLM) поддерживает русский, значит модель должна справляться.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import setup_logging, load_pipeline, generate_music, save_audio, get_vram_usage

logger = setup_logging(__name__, "lesson_10")

def main():
    logger.info("="*60)
    logger.info("Урок 10: Русскоязычная генерация музыки")
    logger.info("="*60)

    # Загрузка модели
    logger.info("Загрузка модели...")
    pipe = load_pipeline()
    logger.info(f"VRAM после загрузки: {get_vram_usage()} MiB")

    output_dir = os.path.expanduser("~/workspace/tmp/learn_minimax-music3/lesson_10")
    os.makedirs(output_dir, exist_ok=True)

    # Тест 1: Простая русская песня
    logger.info("\n[1/3] Простая русская поп-песня...")
    t0 = time.time()
    result = generate_music(
        pipe,
        prompt="Russian pop song, female vocals, upbeat, cheerful",
        lyrics="""[Verse]
Летний вечер, тёплый бриз
Город спит, но мы не спим
Свет фонарей, наш разговор
Снова начинается новый день

[Chorus]
Мы вместе, мы вместе, мы вместе навсегда
Наша музыка звучит, как будто первая
Мы вместе, мы вместе, этот вечер для нас
Пусть музыка играет до рассвета
""",
        duration=20.0,
        seed=42
    )
    wav_path = save_audio(result, os.path.join(output_dir, "russian_pop.wav"))
    logger.info(f"  Сохранено: {wav_path} ({time.time()-t0:.1f}с)")

    # Тест 2: Русский рэп
    logger.info("\n[2/3] Русский рэп...")
    t0 = time.time()
    result = generate_music(
        pipe,
        prompt="Russian rap, male vocals, heavy beat, modern hip-hop",
        lyrics="""[Verse]
Городские огни, ночь на дворе
Мы идём своим путём, не оглядываясь назад
Каждый день — новый вызов, каждый шаг — вперёд
Мы строим свою историю, свой мир

[Chorus]
Это наш город, наша музыка
Мы не останавливаемся, мы не сдаёмся
Это наш город, наша история
Мы пишем её сами, без правил
""",
        duration=20.0,
        seed=123
    )
    wav_path = save_audio(result, os.path.join(output_dir, "russian_rap.wav"))
    logger.info(f"  Сохранено: {wav_path} ({time.time()-t0:.1f}с)")

    # Тест 3: Русская баллада
    logger.info("\n[3/3] Русская баллада...")
    t0 = time.time()
    result = generate_music(
        pipe,
        prompt="Russian ballad, piano, emotional female vocals, sad melody",
        lyrics="""[Verse]
Дождь за окном, и снова ты ушёл
Оставив только тишину и пустоту
Я жду тебя, но ты не возвращаешься
И время летит, как будто никогда

[Chorus]
Почему ты ушёл, почему молчишь
Я так хочу тебя услышать снова
Но тишина отвечает мне
И я одна в этом холодном мире
""",
        duration=20.0,
        seed=456
    )
    wav_path = save_audio(result, os.path.join(output_dir, "russian_ballad.wav"))
    logger.info(f"  Сохранено: {wav_path} ({time.time()-t0:.1f}с)")

    # Конвертация в OGG для отправки
    logger.info("\nКонвертация в OGG...")
    import subprocess
    for wav in ["russian_pop.wav", "russian_rap.wav", "russian_ballad.wav"]:
        ogg = wav.replace(".wav", ".ogg")
        subprocess.run(
            ["ffmpeg", "-i", os.path.join(output_dir, wav),
             "-codec:a", "libvorbis", "-b:a", "128k",
             os.path.join(output_dir, ogg)],
            capture_output=True
        )
        os.remove(os.path.join(output_dir, wav))

    logger.info("\n" + "="*60)
    logger.info("Урок 10 завершён!")
    logger.info(f"Результаты: {output_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()