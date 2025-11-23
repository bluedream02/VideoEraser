# AnimateDiff with VideoEraser

VideoEraser integration with AnimateDiff for concept erasure in animated video generation.

## Quick Start

### Minimal Example

```bash
python scripts/animate.py \
    --prompt "A man running under starry night by Van Gogh." \
    --erased-concept "Van Gogh" \
    --output-dir ./my_videos \
    --seed 42
```

**Note**: `--pretrained-model-path` has a default value (`models/stable-diffusion-v1-5`), so it can be omitted if using the default model.

### With Custom Settings
```bash
python scripts/animate.py \
    --pretrained-model-path models/stable-diffusion-v1-5 \
    --prompt "A man running under starry night by Van Gogh." \
    --erased-concept "Van Gogh" \
    --negative-prompt "blurry, low quality" \
    --output-dir ./outputs \
    --seed 42 \
    --W 512 \
    --H 512 \
    --L 16
```

**Parameters**:
- `--pretrained-model-path`: Path to pretrained Stable Diffusion model (default: `models/stable-diffusion-v1-5`)
- `--prompt`: Text prompt for video generation (required if not using config file)
- `--erased-concept`: Concept to erase from generation (required if not using config file)
- `--output-dir`: Output directory for generated videos (default: `samples/{config}-{timestamp}`)
- `--seed`: Random seed for reproducibility
- `--W`, `--H`: Video width and height (default: 512)
- `--L`: Video length in frames (default: 16)
- `--negative-prompt`: Negative prompt to condition against

