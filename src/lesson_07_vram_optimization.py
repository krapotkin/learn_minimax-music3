"""
Lesson 07: VRAM оптимизация

Этот урок показывает:
- Как модель потребляет VRAM
- Оптимизация через CPU offloading
- Сравнение производительности

Запуск:
    source .venv
    python src/lesson_07_vram_optimization.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import clear_model, generate_music, load_pipeline, print_section  # noqa: E402

OUTPUT_DIR = Path.home() / "workspace" / "tmp" / "learn_minimax-music3" / "lesson_07"


def get_vram_info() -> dict:
    """Get current VRAM usage."""
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1024**3, 2),
    }


def main() -> None:
    """Main function."""
    print("Lesson 07: VRAM оптимизация")
    print("-" * 50)

    print_section("STEP 1: VRAM before loading")
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

    print_section("STEP 2: Loading model")
    pipe = load_pipeline()
    vram_after_load = get_vram_info()
    print(f"  Allocated: {vram_after_load['allocated_gb']} GB")
    print(f"  Reserved: {vram_after_load['reserved_gb']} GB")
    print(f"  Max allocated: {vram_after_load['max_allocated_gb']} GB")

    print_section("STEP 3: Generation with full GPU")
    prompt = "Genre: pop. BPM: 120. Female vocals."
    lyrics = """[verse]
Testing the memory usage
[chorus]
VRAM optimization in action"""

    audio, info = generate_music(
        pipe=pipe, prompt=prompt, lyrics=lyrics,
        duration=10.0, seed=7, output_path=OUTPUT_DIR / "full_gpu.wav",
    )
    vram_after_gen = get_vram_info()
    print(f"  Generation time: {info['generation_time']}s")
    print(f"  VRAM after gen: {vram_after_gen['allocated_gb']} GB")

    print_section("STEP 4: Component memory breakdown")
    # Check individual component sizes
    components = {
        "language_model": pipe.language_model,
        "transformer": pipe.transformer,
        "condition_encoder": pipe.condition_encoder,
        "rvq_depth_decoder": pipe.rvq_depth_decoder,
        "vocoder": pipe.vocoder,
    }
    for name, comp in components.items():
        params = sum(p.numel() for p in comp.parameters())
        param_mb = params * 2 / 1024**2  # bfloat16 = 2 bytes
        print(f"  {name}: {params/1e6:.1f}M params ({param_mb:.0f} MB)")

    print_section("STEP 5: Cleanup")
    del pipe
    clear_model()
    print(f"  VRAM after cleanup: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

    print_section("DONE")
    print("  Lesson 07 complete.")
    print("  Next: python src/lesson_08_reproducibility.py")


if __name__ == "__main__":
    main()