"""
Lesson 11: Длинная генерация (1 и 2 минуты)

Проверяем способность MiniMax Music 3 генерировать длинные треки
на русском языке с сохранением структуры и качества.
"""

import gc
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import setup_logging, load_pipeline, generate_music, save_audio, get_vram_usage

logger = setup_logging(__name__, "lesson_11")

def clear_gpu():
    """Clear GPU memory between generations."""
    gc.collect()
    import torch
    torch.cuda.empty_cache()

def main():
    logger.info("="*60)
    logger.info("Урок 11: Длинная генерация (1 и 2 минуты)")
    logger.info("="*60)

    # Загрузка модели
    logger.info("Загрузка модели...")
    pipe = load_pipeline()
    logger.info(f"VRAM после загрузки: {get_vram_usage()} MiB")

    output_dir = os.path.expanduser("~/workspace/tmp/learn_minimax-music3/lesson_11")
    os.makedirs(output_dir, exist_ok=True)

    # Тест 1: 1 минута — русская поп-песня с полной структурой
    logger.info("\n[1/2] Русская поп-песня, 60 секунд...")
    t0 = time.time()
    result = generate_music(
        pipe,
        prompt="Russian pop song, female vocals, upbeat, cheerful, modern production, 120 BPM",
        lyrics="""[Verse 1]
Летний вечер, тёплый бриз
Город спит, но мы не спим
Свет фонарей, наш разговор
Снова начинается новый день

[Chorus]
Мы вместе, мы вместе, мы вместе навсегда
Наша музыка звучит, как будто первая
Мы вместе, мы вместе, этот вечер для нас
Пусть музыка играет до рассвета

[Verse 2]
Звёзды светят, небо ясно
Мы бежим навстречу счастью
Каждый миг — как первый раз
В этом городе для нас

[Chorus]
Мы вместе, мы вместе, мы вместе навсегда
Наша музыка звучит, как будто первая
Мы вместе, мы вместе, этот вечер для нас
Пусть музыка играет до рассвета

[Bridge]
Небо, звёзды, море, ветер
Мы летим, мы не боимся
Вместе мы — мы сильнее
Наша песня — наша сила

[Chorus]
Мы вместе, мы вместе, мы вместе навсегда
Наша музыка звучит, как будто первая
Мы вместе, мы вместе, этот вечер для нас
Пусть музыка играет до рассвета
""",
        duration=60.0,
        seed=777
    )
    wav_path = save_audio(result, os.path.join(output_dir, "russian_pop_60s.wav"))
    logger.info(f"  Сохранено: {wav_path} ({time.time()-t0:.1f}с)")

    # Очистка памяти между генерациями
    clear_gpu()
    logger.info(f"VRAM после очистки: {get_vram_usage()} MiB")

    # Тест 2: 90 секунд — русская рок-баллада (120с даёт OOM на 24GB VRAM)
    logger.info("\n[2/2] Русская рок-баллада, 90 секунд...")
    t0 = time.time()
    result = generate_music(
        pipe,
        prompt="Russian rock ballad, male vocals, electric guitar, emotional, powerful, 90 BPM",
        lyrics="""[Verse 1]
Дождь стучит по крышам города
Огни мелькают в темноте
Я иду один по улицам
Ищу тебя в этой тишине

[Verse 2]
Время летит, как будто никогда
Ты ушла, но я не сдамся
Каждый день я жду тебя
Хотя не знаю, вернешься ли

[Chorus]
Почему ты ушла, почему молчишь
Я так хочу тебя услышать снова
Но тишина отвечает мне
И я один в этом холодном мире

[Verse 3]
Ветер воет, ночь на дворе
Сердце бьётся, как в первый раз
Я помню каждый миг с тобой
И каждый шаг назад

[Chorus]
Почему ты ушла, почему молчишь
Я так хочу тебя услышать снова
Но тишина отвечает мне
И я один в этом холодном мире

[Bridge]
Может быть, мы встретимся снова
Может быть, это только сон
Но я буду ждать, я буду верить
Что ты вернёшься ко мне

[Guitar Solo]

[Chorus]
Почему ты ушла, почему молчишь
Я так хочу тебя услышать снова
Но тишина отвечает мне
И я один в этом холодном мире

[Outro]
Дождь стучит по крышам города
Я иду один...
И жду тебя...
""",
        duration=90.0,
        seed=888
    )
    wav_path = save_audio(result, os.path.join(output_dir, "russian_rock_90s.wav"))
    logger.info(f"  Сохранено: {wav_path} ({time.time()-t0:.1f}с)")

    # Конвертация в OGG
    logger.info("\nКонвертация в OGG...")
    import subprocess
    for wav in ["russian_pop_60s.wav", "russian_rock_90s.wav"]:
        ogg = wav.replace(".wav", ".ogg")
        subprocess.run(
            ["ffmpeg", "-i", os.path.join(output_dir, wav),
             "-codec:a", "libvorbis", "-b:a", "128k",
             os.path.join(output_dir, ogg)],
            capture_output=True
        )
        os.remove(os.path.join(output_dir, wav))

    logger.info("\n" + "="*60)
    logger.info("Урок 11 завершён!")
    logger.info(f"Результаты: {output_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()