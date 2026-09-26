"""
Lesson 01: Знакомство с моделью MiniMax Music 3

Этот урок показывает:
- Проверку окружения (Python, CUDA, GPU)
- Загрузку модели через ручной загрузчик компонентов
- Параметры модели, архитектура, компоненты
- Измерение потребления памяти

Запуск:
    source .venv
    python src/lesson_01_model_intro.py
"""

import gc
import sys
from pathlib import Path
from typing import Any

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import clear_model, get_model_info, load_pipeline, print_section  # noqa: E402


def check_environment() -> dict[str, Any]:
    """Check Python, CUDA, and GPU environment."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            info[f"gpu_{i}"] = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_memory
            info[f"gpu_{i}_mem_gb"] = round(total / (1024**3), 1)
    return info


def check_diffusers() -> dict[str, Any]:
    """Check diffusers and ModularPipeline availability."""
    import diffusers
    from diffusers import ModularPipeline  # noqa: F401

    return {
        "diffusers_version": diffusers.__version__,
        "modular_pipeline": "available",
    }


def main() -> None:
    """Main function."""
    print("Lesson 01: Знакомство с моделью MiniMax Music 3")
    print("-" * 50)

    # Step 1: Environment
    print_section("STEP 1: Environment check")
    env = check_environment()
    for k, v in env.items():
        print(f"  {k}: {v}")

    # Step 2: diffusers
    print_section("STEP 2: diffusers check")
    diff = check_diffusers()
    for k, v in diff.items():
        print(f"  {k}: {v}")

    # Step 3: Model architecture
    print_section("STEP 3: Model architecture")
    info = get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Step 4: Load model
    print_section("STEP 4: Loading model")
    print("  Loading components...")
    pipe = load_pipeline()
    vram = torch.cuda.memory_allocated(0) / (1024**2)
    print(f"  VRAM allocated: {vram:.0f} MB ({vram/1024:.1f} GB)")

    # Step 5: Cleanup
    print_section("STEP 5: Cleanup")
    del pipe
    clear_model()
    print("  Model cleared from memory")

    print_section("DONE")
    print("  Lesson 01 complete.")
    print("  Next: python src/lesson_02_basic_inference.py")


if __name__ == "__main__":
    main()