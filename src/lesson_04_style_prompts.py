"""
Lesson 04: Детальные описания стиля (Structured Caption)

Этот урок показывает:
- Как детализировать prompt для точного контроля стиля
- BPM, Key, Vocal Details, Arrangement
- Сравнение минимального и детального prompt

Запуск:
    source .venv
    python src/lesson_04_style_prompts.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import clear_model, generate_music, load_pipeline, print_section  # noqa: E402

OUTPUT_DIR = Path.home() / "workspace" / "tmp" / "learn_minimax-music3" / "lesson_04"


def main() -> None:
    """Main function."""
    print("Lesson 04: Детальные описания стиля")
    print("-" * 50)

    print_section("STEP 1: Loading model")
    pipe = load_pipeline()
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

    lyrics = """[verse]
Wandering through the neon glow
City never sleeps, it's time to go
[chorus]
Dancing in the rhythm of the night
Everything will be alright"""

    # Minimal prompt
    print_section("STEP 2: Minimal prompt")
    prompt_minimal = "Pop music"
    audio, info = generate_music(
        pipe=pipe, prompt=prompt_minimal, lyrics=lyrics,
        duration=10.0, seed=7, output_path=OUTPUT_DIR / "minimal_prompt.wav",
    )
    print(f"  Prompt: {prompt_minimal}")
    print(f"  Duration: {info['duration_actual']}s | Time: {info['generation_time']}s")
    print(f"  RMS: {info['rms']} | Peak: {info['peak']}")

    # Detailed prompt
    print_section("STEP 3: Detailed prompt")
    prompt_detailed = (
        "Genre: synth-pop. BPM: 128. Key: C major. "
        "Vocal Details: female, bright, energetic. "
        "Arrangement: synthesizer, drum machine, bass guitar, electronic effects."
    )
    audio, info = generate_music(
        pipe=pipe, prompt=prompt_detailed, lyrics=lyrics,
        duration=10.0, seed=7, output_path=OUTPUT_DIR / "detailed_prompt.wav",
    )
    print(f"  Prompt: {prompt_detailed}")
    print(f"  Duration: {info['duration_actual']}s | Time: {info['generation_time']}s")
    print(f"  RMS: {info['rms']} | Peak: {info['peak']}")

    # Jazz style
    print_section("STEP 4: Jazz style")
    prompt_jazz = (
        "Genre: jazz. BPM: 90. Key: D minor. "
        "Vocal Details: male, smooth, relaxed. "
        "Arrangement: saxophone, piano, upright bass, brushed drums."
    )
    lyrics_jazz = """[verse]
Smooth like velvet in the moonlight
Jazz is playing soft and right
[chorus]
Swing with me through the evening breeze
Saxophone carrying me"""
    audio, info = generate_music(
        pipe=pipe, prompt=prompt_jazz, lyrics=lyrics_jazz,
        duration=10.0, seed=7, output_path=OUTPUT_DIR / "jazz.wav",
    )
    print(f"  Duration: {info['duration_actual']}s | Time: {info['generation_time']}s")
    print(f"  RMS: {info['rms']} | Peak: {info['peak']}")

    # Electronic style
    print_section("STEP 5: Electronic style")
    prompt_electronic = (
        "Genre: electronic dance. BPM: 140. Key: A minor. "
        "Vocal Details: female, processed, ethereal. "
        "Arrangement: synthesizer, kick drum, hi-hat, arpeggiator, reverb."
    )
    lyrics_electronic = """[verse]
Electric pulses through the wire
Digital dreams take me higher
[chorus]
Feel the beat drop low
Let the music take control"""
    audio, info = generate_music(
        pipe=pipe, prompt=prompt_electronic, lyrics=lyrics_electronic,
        duration=10.0, seed=7, output_path=OUTPUT_DIR / "electronic.wav",
    )
    print(f"  Duration: {info['duration_actual']}s | Time: {info['generation_time']}s")
    print(f"  RMS: {info['rms']} | Peak: {info['peak']}")

    print_section("STEP 6: Cleanup")
    del pipe
    clear_model()

    print_section("DONE")
    print("  Lesson 04 complete.")
    print("  Next: python src/lesson_05_duration_control.py")


if __name__ == "__main__":
    main()