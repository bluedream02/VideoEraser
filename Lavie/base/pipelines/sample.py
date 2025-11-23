import os
import torch
import argparse
import torchvision

from pipeline_videogen_original import VideoGenPipeline as VideoGenPipelineOriginal
from pipeline_videogen_removal import VideoGenPipeline as VideoGenPipelineRemoval

from download import find_model
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from transformers import CLIPFeatureExtractor
from omegaconf import OmegaConf

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from models import get_models
import imageio

def main(args):
    import time
    start_time = time.time()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

    # Set to evaluation mode
    unet.eval()
    vae.eval()
    text_encoder_one.eval()

    # Load scheduler based on selected sampling method
    if args.sample_method == 'ddim':
        scheduler = DDIMScheduler.from_pretrained(sd_path, 
                                                subfolder="scheduler",
                                                beta_start=args.beta_start, 
                                                beta_end=args.beta_end, 
                                                beta_schedule=args.beta_schedule)
    elif args.sample_method == 'eulerdiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(sd_path,
                                                        subfolder="scheduler",
                                                        beta_start=args.beta_start,
                                                        beta_end=args.beta_end,
                                                        beta_schedule=args.beta_schedule)
    elif args.sample_method == 'ddpm':
        scheduler = DDPMScheduler.from_pretrained(sd_path,
                                                subfolder="scheduler",
                                                beta_start=args.beta_start,
                                                beta_end=args.beta_end,
                                                beta_schedule=args.beta_schedule)
    else:
        raise NotImplementedError


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Code execution time 0: {elapsed_time:.4f} seconds")

    import time
    start_time = time.time()
    
    # Initialize two Pipelines
    pipeline_original = VideoGenPipelineOriginal(vae=vae, 
                                                 text_encoder=text_encoder_one, 
                                                 tokenizer=tokenizer_one, 
                                                 scheduler=scheduler, 
                                                 unet=unet).to(device)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Code execution time 1: {elapsed_time:.4f} seconds")
    import time
    start_time = time.time()
    pipeline_removal = VideoGenPipelineRemoval(vae=vae, 
                                               text_encoder=text_encoder_one, 
                                               tokenizer=tokenizer_one, 
                                               scheduler=scheduler, 
                                               unet=unet).to(device)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Code execution time 2: {elapsed_time:.4f} seconds")
    

    pipeline_original.enable_xformers_memory_efficient_attention()
    pipeline_removal.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Generate videos
    for prompt in args.text_prompt:
        print(f'Processing the ({prompt}) prompt for both pipelines')


        import time
        start_time = time.time()

        # Use Original Pipeline
        videos_original = pipeline_original(prompt, 
                                            video_length=args.video_length, 
                                            height=args.image_size[0], 
                                            width=args.image_size[1], 
                                            num_inference_steps=args.num_sampling_steps,
                                            guidance_scale=args.guidance_scale).video
        output_path_original = os.path.join(args.output_folder, f'original_{prompt.replace(" ", "_")}_{args.seed}.mp4')
        imageio.mimwrite(output_path_original, videos_original[0], fps=8, quality=9)


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Code execution time 3: {elapsed_time:.4f} seconds")


        import time
        start_time = time.time()

        # Use Removal Pipeline
        videos_removal = pipeline_removal(prompt, 
                                          video_length=args.video_length, 
                                          height=args.image_size[0], 
                                          width=args.image_size[1], 
                                          num_inference_steps=args.num_sampling_steps,
                                          guidance_scale=args.guidance_scale).video
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Code execution time 4: {elapsed_time:.4f} seconds")
        
        output_path_removal = os.path.join(args.output_folder, f'removal_{prompt.replace(" ", "_")}_{args.seed}.mp4')
        imageio.mimwrite(output_path_removal, videos_removal[0], fps=8, quality=9)

        print(f'Videos for prompt "{prompt}" saved: original and removal pipelines')

    print(f'All videos saved in {args.output_folder}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LaVie with VideoEraser: Generate videos with concept erasure")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    
    # Override config parameters
    parser.add_argument("--text-prompt", type=str, default=None, 
                        help="Text prompt for video generation (overrides config)")
    parser.add_argument("--unlearn-prompt", type=str, default=None,
                        help="Concept to erase/unlearn (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for generated videos (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    
    args = parser.parse_args()

    # Load config from YAML
    config = OmegaConf.load(args.config)
    
    # Override config with command-line arguments if provided
    if args.text_prompt is not None:
        config.text_prompt = [args.text_prompt]
    if args.unlearn_prompt is not None:
        config.unlearn_prompt = args.unlearn_prompt
    if args.output_dir is not None:
        config.output_folder = args.output_dir
    if args.seed is not None:
        config.seed = args.seed

    main(config)