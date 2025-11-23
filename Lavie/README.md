# LaVie with VideoEraser

VideoEraser integration with LaVie for concept erasure in text-to-video generation.

## Quick Start

### Basic Usage

```bash
python pipelines/sample.py \
    --config configs/example.yaml \
    --text-prompt "A man running under starry night by Van Gogh." \
    --unlearn-prompt "Van Gogh" \
    --output-dir ./my_videos \
    --seed 42
```

### Using Config File Only

```bash
# Use all settings from config file
python pipelines/sample.py --config configs/example.yaml
```

**Note**: LaVie now supports both command-line arguments and configuration files. Command-line arguments will override config file settings.
