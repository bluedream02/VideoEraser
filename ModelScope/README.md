# ModelScope with VideoEraser

VideoEraser integration with ModelScope/ZeroScope for concept erasure in text-to-video generation.

## Quick Start

```bash
cd ModelScope

python inference.py \
    --model cerspense/zeroscope_v2_576w \
    --prompt "A man running under starry night by Van Gogh." \
    --erased-concept "Van Gogh" \
    --output ./outputs \
    --seed 42
```

**Required Parameters**:
- `--model` (or `-m`): HuggingFace repository or path to model checkpoint directory
- `--prompt` (or `-p`): Text prompt for video generation
- `--erased-concept` (or `-e`): Concept to erase from generation
- `--output` (or `-o`): Directory to save output video to

**Optional Parameters**:
- `--seed` (or `-r`): Random seed for reproducibility
- `--width` (or `-W`): Width of output video (default: 256)
- `--height` (or `-H`): Height of output video (default: 256)
- `--num-frames` (or `-T`): Total number of frames to generate (default: 16)


## Supported Models

This implementation supports both **ZeroScope** and **ModelScope** backbones:

### ZeroScope
```bash
# HuggingFace model ID
python inference.py \
    --model cerspense/zeroscope_v2_576w \
    --prompt "A man running under starry night by Van Gogh." \
    --erased-concept "Van Gogh" \
    --output ./outputs

# Or local path
python inference.py \
    --model models/zeroscope_v2_576w \
    --prompt "A man running under starry night by Van Gogh." \
    --erased-concept "Van Gogh" \
    --output ./outputs
```

### ModelScope
```bash
# HuggingFace model ID
python inference.py \
    --model damo-vilab/text-to-video-ms-1.7b \
    --prompt "A man running under starry night by Van Gogh." \
    --erased-concept "Van Gogh" \
    --output ./outputs

# Or local path
python inference.py \
    --model models/modelscope \
    --prompt "A man running under starry night by Van Gogh." \
    --erased-concept "Van Gogh" \
    --output ./outputs
```
