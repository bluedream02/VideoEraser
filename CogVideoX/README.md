# CogVideoX with VideoEraser

VideoEraser integration with CogVideoX for concept erasure in text-to-video generation.

## Quick Start

```bash
# Use HuggingFace model ID
python cli_demo.py \
    --model_path THUDM/CogVideoX-5b \
    --prompt "A man running under starry night by Van Gogh." \
    --unsafe_concept "Van Gogh" \
    --output_path ./outputs/output.mp4

# Or use local path
python cli_demo.py \
    --model_path models/CogVideoX-5b \
    --prompt "A man running under starry night by Van Gogh." \
    --unsafe_concept "Van Gogh" \
    --output_path ./outputs/output.mp4
```

**Required Parameters**:
- `--prompt`: The description of the video to be generated
- `--unsafe_concept`: The concept to erase (e.g., "Van Gogh", "porn", "violence")

**Optional Parameters**:
- `--model_path`: Path to pre-trained model (default: uses hardcoded path, recommend specifying)
- `--output_path`: Path where the generated video will be saved (default: `./output.mp4`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--guidance_scale`: Scale for classifier-free guidance (default: 6.0)
- `--num_inference_steps`: Number of steps for inference (default: 50)
- `--generate_type`: Type of video generation - `t2v`, `i2v`, or `v2v` (default: `t2v`)
