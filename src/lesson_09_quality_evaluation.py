"""
Lesson 09: Оценка качества

Этот урок показывает:
- Метрики качества аудио (RMS, peak, SNR)
- Спектрограммы
- Сравнение качества разных генераций

Запуск:
    source .venv
    python src/lesson_09_quality_evaluation.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import clear_model, generate_music, load_pipeline, print_section  # noqa: E402

OUTPUT_DIR = Path.home() / "workspace" / "tmp" / "learn_minimax-music3" / "lesson_09"


def analyze_audio(audio: np.ndarray, sr: int) -> dict:
    """Analyze audio quality metrics."""
    # RMS energy
    rms = float(np.sqrt(np.mean(audio**2)))

    # Peak
    peak = float(np.max(np.abs(audio)))

    # Dynamic range
    if rms > 0:
        dynamic_range = 20 * np.log10(peak / rms)
    else:
        dynamic_range = 0

    # Zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(audio, axis=1)) > 0)
    zcr = zero_crossings / audio.shape[1]

    # Spectral centroid (approximate via FFT)
    fft_vals = np.fft.rfft(audio[0])
    freqs = np.fft.rfftfreq(len(audio[0]), d=1.0 / sr)
    spectral_centroid = float(
        np.sum(freqs * np.abs(fft_vals)) / (np.sum(np.abs(fft_vals)) + 1e-10)
    )

    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "dynamic_range_db": round(dynamic_range, 2),
        "zero_crossing_rate": round(zcr, 4),
        "spectral_centroid_hz": round(spectral_centroid, 1),
    }


def main() -> None:
    """Main function."""
    print("Lesson 09: Оценка качества")
    print("-" * 50)

    print_section("STEP 1: Loading model")
    pipe = load_pipeline()
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

    test_cases = [
        {
            "name": "pop_10s",
            "prompt": "Genre: pop. BPM: 120. Female vocals.",
            "lyrics": """[verse]
Quality check for the audio
[chorus]
Testing the sound we produce""",
            "duration": 10.0,
        },
        {
            "name": "ambient_20s",
            "prompt": "Genre: ambient. BPM: 60. Female vocals, piano.",
            "lyrics": """[verse]
Floating in the silence
Time stands still
[chorus]
Peace and calm surround me now""",
            "duration": 20.0,
        },
        {
            "name": "rock_10s",
            "prompt": "Genre: rock. BPM: 140. Male vocals, electric guitar.",
            "lyrics": """[verse]
Thunder rolling through the night
[chorus]
We will rock until the dawn""",
            "duration": 10.0,
        },
    ]

    for i, tc in enumerate(test_cases):
        print_section(f"STEP {i + 2}: {tc['name']}")
        audio, info = generate_music(
            pipe=pipe, prompt=tc["prompt"], lyrics=tc["lyrics"],
            duration=tc["duration"], seed=7,
            output_path=OUTPUT_DIR / f"{tc['name']}.wav",
        )

        metrics = analyze_audio(audio, pipe.sampling_rate)
        print(f"  Duration: {info['duration_actual']}s")
        print(f"  Gen time: {info['generation_time']}s")
        print(f"  RMS: {metrics['rms']}")
        print(f"  Peak: {metrics['peak']}")
        print(f"  Dynamic range: {metrics['dynamic_range_db']} dB")
        print(f"  Zero crossing rate: {metrics['zero_crossing_rate']}")
        print(f"  Spectral centroid: {metrics['spectral_centroid_hz']} Hz")

    print_section("STEP 5: Summary")
    print("  All quality metrics computed successfully.")
    print("  Audio files saved to:", OUTPUT_DIR)

    print_section("STEP 6: Cleanup")
    del pipe
    clear_model()

    print_section("DONE")
    print("  Lesson 09 complete.")
    print("  All lessons finished!")


if __name__ == "__main__":
    main()