"""
Lesson 05: Контроль длительности

Этот урок показывает:
- Как параметр audio_duration влияет на результат
- Сравнение коротких и длинных генераций
- Максимальная длительность (300 сек)

Запуск:
    source .venv
    python src/lesson_05_duration_control.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import clear_model, generate_music, load_pipeline, print_section  # noqa: E402

OUTPUT_DIR = Path.home() / "workspace" / "tmp" / "learn_minimax-music3" / "lesson_05"


def main() -> None:
    """Main function."""
    print("Lesson 05: Контроль длительности")
    print("-" * 50)

    print_section("STEP 1: Loading model")
    pipe = load_pipeline()
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

    prompt = "Genre: ambient. BPM: 60. Female vocals, piano, strings."
    lyrics = """[verse]
Floating in the silence
Time stands still
[chorus]
Peace and calm surround me now
In this moment I am free"""

    durations = [10.0, 20.0, 30.0]
    for dur in durations:
        print_section(f"STEP {durations.index(dur) + 2}: Duration {dur}s")
        audio, info = generate_music(
            pipe=pipe, prompt=prompt, lyrics=lyrics,
            duration=dur, seed=7, output_path=OUTPUT_DIR / f"duration_{int(dur)}s.wav",
        )
        print(f"  Requested: {info['duration_requested']}s")
        print(f"  Actual: {info['duration_actual']}s")
        print(f"  Generation time: {info['generation_time']}s")
        print(f"  Samples: {info['shape']}")
        print(f"  RMS: {info['rms']} | Peak: {info['peak']}")

    print_section("STEP 5: Cleanup")
    del pipe
    clear_model()

    print_section("DONE")
    print("  Lesson 05 complete.")
    print("  Next: python src/lesson_06_multi_genre.py")


if __name__ == "__main__":
    main()