"""
Lesson 03: Управление структурой через теги секций

Этот урок показывает:
- Как теги секций в lyrics влияют на структуру песни
- Разные форматы тегов: [verse], [chorus], [bridge], [outro]
- Влияние порядка секций на результат

Запуск:
    source .venv
    python src/lesson_03_lyrics_control.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import clear_model, generate_music, load_pipeline, print_section  # noqa: E402

OUTPUT_DIR = Path.home() / "workspace" / "tmp" / "learn_minimax-music3" / "lesson_03"


def main() -> None:
    """Main function."""
    print("Lesson 03: Управление структурой через теги секций")
    print("-" * 50)

    print_section("STEP 1: Loading model")
    pipe = load_pipeline()
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

    # Full song structure
    print_section("STEP 2: Full song structure")
    prompt = "Genre: pop ballad. BPM: 80. Female vocals, piano."
    lyrics_full = """[verse]
Walking alone through the empty street
Shadows dancing to the rhythm of my feet
[chorus]
Oh this melody carries me away
Every note is a brand new day
[verse]
Stars above are shining so bright
Guiding me through the darkest night
[chorus]
Oh this melody carries me away
Every note is a brand new day
[outro]
Fade away... fade away..."""

    audio, info = generate_music(
        pipe=pipe, prompt=prompt, lyrics=lyrics_full,
        duration=20.0, seed=7, output_path=OUTPUT_DIR / "full_structure.wav",
    )
    print(f"  Duration: {info['duration_actual']}s | Time: {info['generation_time']}s")
    print(f"  RMS: {info['rms']} | Peak: {info['peak']}")

    # Verse only
    print_section("STEP 3: Verse only")
    lyrics_verse = """[verse]
Walking alone through the empty street
Shadows dancing to the rhythm of my feet
The city lights are calling my name
Nothing will ever be the same"""

    audio, info = generate_music(
        pipe=pipe, prompt=prompt, lyrics=lyrics_verse,
        duration=10.0, seed=7, output_path=OUTPUT_DIR / "verse_only.wav",
    )
    print(f"  Duration: {info['duration_actual']}s | Time: {info['generation_time']}s")
    print(f"  RMS: {info['rms']} | Peak: {info['peak']}")

    # Chorus only
    print_section("STEP 4: Chorus only")
    lyrics_chorus = """[chorus]
Oh this melody carries me away
Every note is a brand new day
I can feel the music in my soul
It's taking me places I've never known"""

    audio, info = generate_music(
        pipe=pipe, prompt=prompt, lyrics=lyrics_chorus,
        duration=10.0, seed=7, output_path=OUTPUT_DIR / "chorus_only.wav",
    )
    print(f"  Duration: {info['duration_actual']}s | Time: {info['generation_time']}s")
    print(f"  RMS: {info['rms']} | Peak: {info['peak']}")

    # Bridge
    print_section("STEP 5: Bridge section")
    lyrics_bridge = """[bridge]
And when the morning comes
I'll find my way home
Through the silence and the sound
I'll stand my ground"""

    audio, info = generate_music(
        pipe=pipe, prompt=prompt, lyrics=lyrics_bridge,
        duration=10.0, seed=7, output_path=OUTPUT_DIR / "bridge.wav",
    )
    print(f"  Duration: {info['duration_actual']}s | Time: {info['generation_time']}s")
    print(f"  RMS: {info['rms']} | Peak: {info['peak']}")

    print_section("STEP 6: Cleanup")
    del pipe
    clear_model()

    print_section("DONE")
    print("  Lesson 03 complete.")
    print("  Next: python src/lesson_04_style_prompts.py")


if __name__ == "__main__":
    main()