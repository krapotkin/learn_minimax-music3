"""
Lesson 08: Воспроизводимость (Reproducibility)

Этот урок показывает:
- Детерминированность генерации с одинаковым seed
- Сравнение аудио отпечатков
- Влияние seed на результат

Запуск:
    source .venv
    python src/lesson_08_reproducibility.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import clear_model, generate_music, load_pipeline, print_section  # noqa: E402

OUTPUT_DIR = Path.home() / "workspace" / "tmp" / "learn_minimax-music3" / "lesson_08"


def audio_fingerprint(audio: np.ndarray) -> str:
    """Generate a simple fingerprint for audio comparison."""
    # Use hash of first 1000 samples
    samples = audio.flatten()[:1000].tobytes()
    import hashlib

    return hashlib.md5(samples).hexdigest()


def main() -> None:
    """Main function."""
    print("Lesson 08: Воспроизводимость")
    print("-" * 50)

    print_section("STEP 1: Loading model")
    pipe = load_pipeline()
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

    prompt = "Genre: pop. BPM: 120. Female vocals."
    lyrics = """[verse]
Testing reproducibility
Same seed same result
[chorus]
Music flows the same way
Every single time we play"""

    # Generate with same seed twice
    print_section("STEP 2: Same seed, two generations")
    audio1, info1 = generate_music(
        pipe=pipe, prompt=prompt, lyrics=lyrics,
        duration=10.0, seed=42, output_path=OUTPUT_DIR / "seed42_run1.wav",
    )
    fp1 = audio_fingerprint(audio1)
    print(f"  Run 1 fingerprint: {fp1}")

    audio2, info2 = generate_music(
        pipe=pipe, prompt=prompt, lyrics=lyrics,
        duration=10.0, seed=42, output_path=OUTPUT_DIR / "seed42_run2.wav",
    )
    fp2 = audio_fingerprint(audio2)
    print(f"  Run 2 fingerprint: {fp2}")

    # Compare
    diff = np.max(np.abs(audio1 - audio2))
    print(f"  Max difference: {diff:.10f}")
    print(f"  Fingerprints match: {fp1 == fp2}")

    # Different seed
    print_section("STEP 3: Different seed")
    audio3, info3 = generate_music(
        pipe=pipe, prompt=prompt, lyrics=lyrics,
        duration=10.0, seed=99, output_path=OUTPUT_DIR / "seed99.wav",
    )
    fp3 = audio_fingerprint(audio3)
    print(f"  Seed 99 fingerprint: {fp3}")
    print(f"  Same as seed 42: {fp3 == fp1}")

    diff_42_99 = np.max(np.abs(audio1 - audio3))
    print(f"  Max diff (42 vs 99): {diff_42_99:.6f}")

    print_section("STEP 4: Cleanup")
    del pipe
    clear_model()

    print_section("DONE")
    print("  Lesson 08 complete.")
    print("  Next: python src/lesson_09_quality_evaluation.py")


if __name__ == "__main__":
    main()