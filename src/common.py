"""
Common module for MiniMax Music 3 model loading and inference.

Working approach: load each component manually and register with pipeline.
from_pretrained() hangs on network calls for missing config files.
"""

import gc
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

# Fix CUDA memory fragmentation for multi-generation sessions
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LOCAL_MODEL_PATH = str(Path.home() / "workspace" / "models" / "models_music" / "MiniMax-Music3")


def load_pipeline() -> Any:
    """Load MiniMax Music 3 pipeline with all components on GPU.

    Returns:
        Configured MiniMaxMusic3ModularPipeline ready for inference.

    Raises:
        RuntimeError: If model files are not found at LOCAL_MODEL_PATH.
    """
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3RVQDepthDecoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )
    from diffusers.modular_pipelines.minimax_music3.modular_blocks_minimax_music3 import (
        MiniMaxMusic3Blocks,
    )
    from diffusers.modular_pipelines.minimax_music3.modular_pipeline import (
        MiniMaxMusic3ModularPipeline,
    )
    from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

    mp = LOCAL_MODEL_PATH

    tokenizer = Qwen2Tokenizer.from_pretrained(f"{mp}/tokenizer", local_files_only=True)

    language_model = Qwen3ForCausalLM.from_pretrained(
        f"{mp}/language_model",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    ).cuda()

    condition_encoder = MiniMaxMusic3ConditionEncoder.from_pretrained(
        f"{mp}/condition_encoder",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda()

    rvq_depth_decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        f"{mp}/rvq_depth_decoder",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda()

    transformer = MiniMaxMusic3Transformer1DModel.from_pretrained(
        f"{mp}/transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        f"{mp}/scheduler",
        local_files_only=True,
    )

    vocoder = MiniMaxMusic3Vocoder.from_pretrained(
        f"{mp}/vocoder",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda()

    blocks = MiniMaxMusic3Blocks()
    pipe = MiniMaxMusic3ModularPipeline(blocks=blocks)
    pipe.register_components(
        tokenizer=tokenizer,
        language_model=language_model,
        rvq_depth_decoder=rvq_depth_decoder,
        condition_encoder=condition_encoder,
        transformer=transformer,
        scheduler=scheduler,
        vocoder=vocoder,
    )

    return pipe


def generate_music(
    pipe: Any,
    prompt: str,
    lyrics: str,
    duration: float = 20.0,
    seed: int = 7,
    output_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate music with the given prompt and lyrics.

    Args:
        pipe: Loaded MiniMaxMusic3ModularPipeline.
        prompt: Style description (genre, BPM, vocals, arrangement).
        lyrics: Song lyrics with optional section tags.
        duration: Desired audio duration in seconds.
        seed: Random seed for reproducibility.
        output_path: Optional path to save WAV file.

    Returns:
        Tuple of (audio numpy array, info dict).
    """
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    generator = torch.Generator("cuda").manual_seed(seed)
    start = time.time()

    result = pipe(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=duration,
        generator=generator,
        output="audios",
    )
    gen_time = time.time() - start

    audio = result[0]  # numpy array (channels, samples)
    sr = pipe.sampling_rate
    actual_duration = audio.shape[1] / sr

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio.T, sr)

    info = {
        "duration_requested": duration,
        "duration_actual": round(actual_duration, 2),
        "generation_time": round(gen_time, 2),
        "shape": audio.shape,
        "sampling_rate": sr,
        "rms": round(float(np.sqrt(np.mean(audio**2))), 6),
        "peak": round(float(np.max(np.abs(audio))), 6),
    }

    return audio, info


def get_model_info() -> dict[str, Any]:
    """Return model architecture information."""
    return {
        "model_name": "MiniMaxAI/MiniMax-Music3",
        "framework": "diffusers 0.40.0",
        "architecture": "Hybrid-LM + Flow Matching + Flow-VAE",
        "global_llm": "Qwen3-based, 8B params, 36 layers, hidden=4096",
        "local_llm": "AbabForCausalLM, 0.6B params, 36 layers, hidden=4096",
        "flow_matching": "2.4B params",
        "flow_vae": "123M params",
        "tokenizer": "RVQ 8 layers (1 semantic x16384 + 7 acoustic x1024)",
        "output_format": "WAV, 44100 Hz, 16-bit stereo",
        "max_duration_seconds": 300,
        "frames_per_second": 25,
        "vram_required_gb": 22,
        "local_path": LOCAL_MODEL_PATH,
    }


def clear_model() -> None:
    """Clear GPU memory."""
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def setup_logging(name: str, log_file: str = None) -> logging.Logger:
    """Setup logging for a module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def save_audio(audio_result, output_path: str) -> str:
    """Save audio from generation result to WAV file."""
    if isinstance(audio_result, tuple):
        audio_tensor = audio_result[0]
    else:
        audio_tensor = audio_result

    # Extract audio array
    if hasattr(audio_tensor, 'numpy'):
        audio_np = audio_tensor.cpu().numpy()
    else:
        audio_np = np.array(audio_tensor)

    # Handle batch dimension
    if audio_np.ndim > 2:
        audio_np = audio_np[0]  # Take first batch item

    # Ensure stereo (channels, samples) -> (samples, channels)
    if audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T

    sf.write(output_path, audio_np, samplerate=44100)
    return output_path


def get_vram_usage() -> int:
    """Get current VRAM usage in MiB."""
    return torch.cuda.memory_allocated() // (1024 * 1024)