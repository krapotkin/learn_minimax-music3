"""
Lesson 06: Генерация в разных жанрах

Этот урок показывает:
- Генерацию музыки в различных жанрах
- Сравнение качества и характеристик

Запуск:
    source .venv
    python src/lesson_06_multi_genre.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import clear_model, generate_music, load_pipeline, print_section  # noqa: E402

OUTPUT_DIR = Path.home() / "workspace" / "tmp" / "learn_minimax-music3" / "lesson_06"

GENRES = [
    {
        "name": "classical",
        "prompt": "Genre: classical. BPM: 72. Key: C major. Arrangement: piano solo, orchestral strings.",
        "lyrics": """[verse]
A gentle melody flows through the hall
Echoes of time answering the call
[chorus]
Harmony of the spheres above
Classical music, eternal love""",
    },
    {
        "name": "hiphop",
        "prompt": "Genre: hip hop. BPM: 90. Key: G minor. Vocal Details: male, rhythmic. Arrangement: drum machine, bass, samples.",
        "lyrics": """[verse]
Step by step I rise above
Breaking through the walls of doubt
[chorus]
This is my time to shine
Making history every day""",
    },
    {
        "name": "country",
        "prompt": "Genre: country. BPM: 110. Key: G major. Vocal Details: male, warm. Arrangement: acoustic guitar, fiddle, banjo.",
        "lyrics": """[verse]
Down on the farm where the sun shines bright
Everything is golden in the morning light
[chorus]
Country roads take me home
To the place I call my own""",
    },
    {
        "name": "rnb",
        "prompt": "Genre: R&B. BPM: 85. Key: A minor. Vocal Details: female, soulful. Arrangement: keyboard, drums, bass, strings.",
        "lyrics": """[verse]
Your love is like a river flowing
Carrying me where I need to go
[chorus]
Hold me close and never let me go
Together we can make it so""",
    },
    {
        "name": "metal",
        "prompt": "Genre: metal. BPM: 160. Key: E minor. Vocal Details: male, powerful. Arrangement: electric guitar, drums, bass.",
        "lyrics": """[verse]
Thunder crashes through the sky
Fire burning deep inside
[chorus]
Rise up and fight the night
Metal power takes flight""",
    },
]


def main() -> None:
    """Main function."""
    print("Lesson 06: Генерация в разных жанрах")
    print("-" * 50)

    print_section("STEP 1: Loading model")
    pipe = load_pipeline()
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

    for i, genre in enumerate(GENRES):
        print_section(f"STEP {i + 2}: {genre['name'].upper()}")
        audio, info = generate_music(
            pipe=pipe, prompt=genre["prompt"], lyrics=genre["lyrics"],
            duration=10.0, seed=7, output_path=OUTPUT_DIR / f"{genre['name']}.wav",
        )
        print(f"  Genre: {genre['name']}")
        print(f"  Duration: {info['duration_actual']}s | Time: {info['generation_time']}s")
        print(f"  RMS: {info['rms']} | Peak: {info['peak']}")

    print_section("STEP 7: Cleanup")
    del pipe
    clear_model()

    print_section("DONE")
    print("  Lesson 06 complete.")
    print("  Next: python src/lesson_07_vram_optimization.py")


if __name__ == "__main__":
    main()