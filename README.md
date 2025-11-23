<div align="center">

# VideoEraser: Concept Erasure in Text-to-Video Diffusion Models

[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue.svg)](https://2025.emnlp.org/) [![arXiv](https://img.shields.io/badge/arXiv-2508.15314-b31b1b.svg)](https://arxiv.org/abs/2508.15314)

</div>


## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Acknowledgement](#-acknowledgement)
- [Citation](#-citation)


## 📌 Overview

**VideoEraser** is a **training-free** framework that prevents Text-to-Video (T2V) diffusion models from generating videos with undesirable concepts, even when explicitly prompted. It achieves **state-of-the-art** performance in suppressing undesirable content during T2V generation, reducing it by **46% on average** across four tasks compared to baselines.

Besides, VideoEraser is applicable to multiple T2V diffusion models, including UNet-based models – **AnimateDiff**, **LaVie**, **ZeroScope**, **ModelScope**, and a DiT-based model – **CogVideoX**.

### Video Demonstrations

<table width="80%">
<thead>
<tr>
<th></th>
<th>Object Erasure</th>
<th>Celebrity Erasure</th>
<th>Artistic Style Erasure</th>
<th>Explicit Content Erasure</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Task</strong></td>
<td>Object Erasure</td>
<td>Celebrity Erasure</td>
<td>Artistic Style Erasure</td>
<td>Explicit Content Erasure</td>
</tr>
<tr>
<td><strong>Original Prompt</strong></td>
<td>A rally car racing through a snowy forest path.</td>
<td>Jackie Chan is tai chi.</td>
<td>A man running under starry night by Van Gogh.</td>
<td>A naked man is playing basketball.</td>
</tr>
<tr>
<td><strong>Original Video</strong></td>
<td><video src="assets/object.gif" width="160" loop autoplay muted playsinline></video></td>
<td><video src="assets/celebrity.gif" width="160" loop autoplay muted playsinline></video></td>
<td><video src="assets/artist.gif" width="160" loop autoplay muted playsinline></video></td>
<td><video src="assets/explicit.gif" width="160" loop autoplay muted playsinline></video></td>
</tr>
<tr>
<td><strong>Erased Concept</strong></td>
<td>rally car</td>
<td>Jackie Chan</td>
<td>Van Gogh</td>
<td>naked</td>
</tr>
<tr>
<td><strong>Erased Video</strong></td>
<td><video src="assets/object_removal.gif" width="160" loop autoplay muted playsinline></video></td>
<td><video src="assets/celebrity_removal.gif" width="160" loop autoplay muted playsinline></video></td>
<td><video src="assets/artist_removal.gif" width="160" loop autoplay muted playsinline></video></td>
<td><video src="assets/explicit_removal.gif" width="160" loop autoplay muted playsinline></video></td>
</tr>
</tbody>
</table>



## 📢 News

- [2025.11] 🎉 Our paper **"VideoEraser: Concept Erasure in Text-to-Video Diffusion Models"** has been **accepted to EMNLP 2025 Main Conference**!  



## 🔧 Installation


### Setup

#### Option 1: AnimateDiff

```bash
git clone https://github.com/bluedream02/VideoEraser.git
cd VideoEraser/AnimateDiff

# Create environment
conda create -n animatediff python=3.10
conda activate animatediff
pip install -r requirements.txt
```

**Download Pre-trained Models:**

```bash
# Option A: Use HuggingFace model ID (recommended, no manual download needed)
python scripts/animate.py --pretrained-model-path stable-diffusion-v1-5/stable-diffusion-v1-5

# Option B: Download models manually
mkdir -p models/Motion_Module
cd models

# Download Stable Diffusion v1-5
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5

# Download Motion Module
cd Motion_Module
wget https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt
cd ../..
```

Expected structure:
```
AnimateDiff/
├── models/
│   ├── stable-diffusion-v1-5/        # Stable Diffusion base model
│   │   ├── ...
│   └── Motion_Module/
│       └── mm_sd_v15.ckpt            # AnimateDiff motion module
```

#### Option 2: ModelScope (ZeroScope/ModelScope)

```bash
cd VideoEraser/ModelScope

# Create environment
conda create -n modelscope python=3.10
conda activate modelscope
pip install -r requirements.txt
```

**Download Pre-trained Models:**

```bash
# Download models manually
mkdir -p models
cd models

# Download ZeroScope
git lfs install
git clone https://huggingface.co/cerspense/zeroscope_v2_576w

# Or download ModelScope
git clone https://huggingface.co/damo-vilab/text-to-video-ms-1.7b
cd ..
```

Expected structure:
```
ModelScope/
├── models/
│   ├── zeroscope_v2_576w/            # ZeroScope model
│   │   ├── ...
│   └── text-to-video-ms-1.7b/        # ModelScope model (alternative)
│       ├── ...
```

#### Option 3: LaVie

```bash
cd VideoEraser/Lavie

# Create environment
conda env create -f environment.yml
conda activate lavie
```

**Download Pre-trained Models:**

Download pre-trained LaVie models, Stable Diffusion 1.4, and stable-diffusion-x4-upscaler:

```bash
mkdir -p pretrained_models
cd pretrained_models

# Download LaVie checkpoints
wget https://huggingface.co/Vchitect/LaVie/resolve/main/lavie_base.pt
wget https://huggingface.co/Vchitect/LaVie/resolve/main/lavie_interpolation.pt
wget https://huggingface.co/Vchitect/LaVie/resolve/main/lavie_vsr.pt

# Download Stable Diffusion v1-4
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4

# Download Stable Diffusion x4 Upscaler
git clone https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
cd ..
```

Expected structure:
```
Lavie/
├── pretrained_models/
│   ├── lavie_base.pt                 # Base T2V model
│   ├── lavie_interpolation.pt        # Frame interpolation model
│   ├── lavie_vsr.pt                  # Video super-resolution model
│   ├── stable-diffusion-v1-4/        # SD 1.4 base model
│   │   ├── ...
│   └── stable-diffusion-x4-upscaler/ # SD x4 upscaler
│       ├── ...
```

#### Option 4: CogVideoX

```bash
cd VideoEraser/CogVideoX

# Create environment
conda create -n cogvideox python=3.10
conda activate cogvideox
pip install -r requirements.txt
```

**Download Pre-trained Models:**

```bash
# Download models manually
mkdir -p models
cd models

#  download CogVideoX-5b
git clone https://huggingface.co/THUDM/CogVideoX-5b
cd ..
```

Expected structure:
```
CogVideoX/
├── models/
│   ├── CogVideoX-2b/                 # 2B parameters (recommended for testing)
│   │   ├── ...
│   └── CogVideoX-5b/                 # 5B parameters (best quality)
│       ├── ...
```


## 🚀 Quick Start

### AnimateDiff (UNet-based)

```bash
cd AnimateDiff

python scripts/animate.py \
    --pretrained-model-path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --prompt "A man running under starry night by Van Gogh." \
    --erased-concept "Van Gogh" \
    --output-dir ./outputs \
    --seed 42
```

See [AnimateDiff/README.md](AnimateDiff/README.md) for detailed usage.

### ModelScope (UNet-based, ZeroScope/ModelScope)

```bash
cd ModelScope

# Simple usage with HuggingFace model
python inference.py \
    --model cerspense/zeroscope_v2_576w \
    --prompt "A man running under starry night by Van Gogh." \
    --erased-concept "Van Gogh" \
    --output ./outputs \
    --seed 42

# Or with ModelScope backbone
python inference.py \
    --model damo-vilab/text-to-video-ms-1.7b \
    --prompt "A man running under starry night by Van Gogh." \
    --erased-concept "Van Gogh" \
    --output ./outputs

# Using short form arguments
python inference.py \
    -m cerspense/zeroscope_v2_576w \
    -p "A man running under starry night by Van Gogh." \
    -e "Van Gogh" \
    -o ./outputs \
    -r 42
```

See [ModelScope/README.md](ModelScope/README.md) for detailed usage.

### LaVie (UNet-based)

```bash
cd Lavie/base

# Command-line usage (recommended)
python pipelines/sample.py \
    --config configs/example.yaml \
    --text-prompt "A man running under starry night by Van Gogh." \
    --unlearn-prompt "Van Gogh" \
    --output-dir ./outputs \
    --seed 42
```

**Note**: LaVie now supports command-line arguments that override config file settings.

See [Lavie/README.md](Lavie/README.md) for detailed usage.

### CogVideoX (DiT-based)

```bash
cd CogVideoX

# Simple CLI usage (note: uses --unsafe_concept)
python cli_demo.py \
    --prompt "A man running under starry night by Van Gogh." \
    --unsafe_concept "Van Gogh" \
    --model_path THUDM/CogVideoX-2b \
    --output_path ./output.mp4
```

See [CogVideoX/README.md](CogVideoX/README.md) for detailed usage.

### Evaluation

We provide evaluation scripts for assessing concept erasure performance. The scripts process videos frame-by-frame: if any frame contains the target concept, the video is considered to contain that concept.

```bash
cd evaluation

# 1. Artistic Style Detection (requires OpenAI API)
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to OpenAI
python artist.py \
    --input-folder /path/to/videos \
    --output-folder ./results \
    --num-samples 5

# 2. Object Detection
python object.py \
    --input-folder /path/to/videos \
    --output-folder ./results \
    --target-objects cassette player \
    --num-samples 5

# 3. Explicit Content Detection (requires nudenet)
pip install nudenet
python explict.py \
    --input-folder /path/to/videos \
    --output-folder ./results \
    --num-samples 5
```



## 🙏 Acknowledgement

This work builds upon several excellent open-source projects:

- [AnimateDiff](https://github.com/guoyww/AnimateDiff) - Motion module for Stable Diffusion
- [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning) - ZeroScope and ModelScope training framework
- [LaVie](https://github.com/Vchitect/LaVie) - Video generation with cascaded diffusion models
- [CogVideoX](https://github.com/THUDM/CogVideo) - Large-scale text-to-video generation model
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) - Foundation text-to-image model
- [SEGA](https://github.com/ml-research/semantic-image-editing) - Instructing Text-to-Image Models using Semantic Guidance
- [SAFREE](https://github.com/codegoat24/SAFREE) - Safe and free text-to-image generation


We thank the authors for their valuable contributions to the community.



## 📖 Citation

If you find VideoEraser useful in your research, please cite:

```bibtex
@inproceedings{xu2025videoeraser,
  title={VideoEraser: Concept Erasure in Text-to-Video Diffusion Models},
  author={Xu, Naen and Zhang, Jinghuai and Li, Changjiang and Chen, Zhi and Zhou, Chunyi and Li, Qingming and Du, Tianyu and Ji, Shouling},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={5965--5994},
  year={2025}
}
```
