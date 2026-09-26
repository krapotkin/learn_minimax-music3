#!/usr/bin/env python3
"""Скачивание модели MiniMax Music 3 с HuggingFace."""

import argparse
import sys
from pathlib import Path

MODEL_ID = "MiniMaxAI/MiniMax-Music3"
DEFAULT_LOCAL_DIR = str(Path.home() / "workspace" / "models" / "models_music" / "MiniMax-Music3")


def download_model(local_dir: str, resume: bool = True) -> None:
    """Скачивание модели."""
    from huggingface_hub import snapshot_download

    output_path = Path(local_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Скачивание модели {MODEL_ID}...")
    print(f"Целевая директория: {output_path}")
    print()

    try:
        result = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=local_dir,
            resume_download=resume,
            local_dir_use_symlinks=False,
        )
        print(f"Модель успешно скачана в: {result}")
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        print()
        print("Советы:")
        print("  1. Убедитесь, что есть ~60 GB свободного места")
        print("  2. Попробуйте с HF_TOKEN для более высоких лимитов:")
        print("     huggingface-cli login")
        print("  3. Используйте зеркало для ускорения:")
        print("     export HF_ENDPOINT=https://hf-mirror.com")
        print("  4. Используйте hf-transfer для ускорения:")
        print("     pip install hf-transfer")
        print("     export HF_HUB_ENABLE_HF_TRANSFER=1")
        sys.exit(1)


def check_model(local_dir: str) -> None:
    """Проверка наличия модели."""
    model_path = Path(local_dir)

    if not model_path.exists():
        print(f"Модель не найдена в: {model_path}")
        return False

    required_files = [
        "config.json",
        "language_model/config.json",
        "language_model/model.safetensors.index.json",
        "transformer/config.json",
        "vocoder/config.json",
        "tokenizer/tokenizer.json",
    ]

    missing = []
    for f in required_files:
        if not (model_path / f).exists():
            missing.append(f)

    if missing:
        print(f"Модель найдена, но отсутствуют файлы: {missing}")
        return False

    print(f"Модель найдена в: {model_path}")
    print("Все необходимые файлы присутствуют.")
    return True


def main() -> int:
    """Основная функция."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help=f"Путь для сохранения модели (по умолчанию: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument("--check", action="store_true", help="Только проверить наличие модели")
    parser.add_argument("--no-resume", action="store_true", help="Не использовать докачку")

    args = parser.parse_args()

    if args.check:
        check_model(args.local_dir)
        return 0

    # Проверка перед скачиванием
    if Path(args.local_dir).exists():
        print(f"Директория уже существует: {args.local_dir}")
        print("Если модель уже скачана, используйте --check для проверки.")
        response = input("Перезаписать? (y/N): ")
        if response.lower() != "y":
            print("Отмена.")
            return 0

    download_model(args.local_dir, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())