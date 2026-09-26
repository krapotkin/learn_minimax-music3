# AGENTS.md - MiniMax Music 3

## Project Overview

Educational project for MiniMax Music 3 music generation model. 9 lessons covering model architecture, inference, lyrics control, style prompts, duration, genres, VRAM optimization, reproducibility, and quality evaluation.

## Key Facts

- **Model**: MiniMaxAI/MiniMax-Music3 (~54 GB)
- **Local path**: `~/workspace/models/models_music/MiniMax-Music3/`
- **venv**: `~/workspace/venvs/learn_minimax-music3/cuda_12_4/`
- **Python**: 3.12.13
- **CUDA**: 12.4
- **GPU**: RTX 3090 (24 GB VRAM)
- **VRAM usage**: ~22.4 GB
- **diffusers**: 0.40.0
- **torch**: 2.6.0+cu124

## Critical: Model Loading

**DO NOT use `ModularPipeline.from_pretrained()`** - it hangs on network calls for missing config files.

**CORRECT approach** (in `src/common.py`):
1. Load each component manually with `local_files_only=True`
2. Move to CUDA manually
3. Register with pipeline via `register_components()`

```python
from src.common import load_pipeline, generate_music
pipe = load_pipeline()
audio, info = generate_music(pipe, prompt, lyrics, duration=10.0, seed=7)
```

## Critical: Memory Management

- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already in `common.py`)
- Call `gc.collect()` and `torch.cuda.empty_cache()` before each generation
- Model occupies ~22.4 GB VRAM - leave ~1.6 GB for vocoder during inference

## File Structure

- `src/common.py` - Shared module: `load_pipeline()`, `generate_music()`, `clear_model()`
- `src/lesson_01` through `lesson_09` - Educational scripts
- `scripts/download_model.py` - Model download script
- `docs/architecture.md` - Model architecture details
- `docs/style_guide.md` - Prompt writing guide

## Output

Audio files saved to `~/workspace/tmp/learn_minimax-music3/lesson_XX/`.

## Known Issues

1. `from_pretrained()` hangs on network - use manual loading
2. OOM without `expandable_segments:True` - already handled in common.py
3. `modular_config.json` must exist in model directory - already created
4. `trust_remote_code=True` required for Qwen3ForCausalLM