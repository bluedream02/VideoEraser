import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification
from typing import List, Optional, Set
from tqdm import tqdm


def sample_video_frames(video_path: str, num_samples: int = 5) -> List:
    """
    Sample multiple frames from video
    
    Args:
        video_path: Path to video file
        num_samples: Number of frames to sample
    
    Returns:
        List of sampled frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    
    # Uniformly sample frames
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    frames = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def classify_frame(processor, model, frame, target_labels: Optional[List[str]] = None):
    """
    Process single frame image and return classification result
    
    Args:
        processor: Image processor
        model: Classification model
        frame: Video frame (BGR format)
        target_labels: List of target labels, if provided then only check these labels
    
    Returns:
        If target_labels is None, return top-5 prediction results
        If target_labels is not None, return whether target labels are detected
    """
    # Convert frame to PIL image and ensure RGB format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).convert("RGB")
    
    # Process image to fit model input
    inputs = processor(image, return_tensors="pt")
    
    # Inference phase (disable gradient computation to save memory)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Convert to probability distribution
    probabilities = torch.softmax(logits, dim=-1)[0]
    
    if target_labels is not None:
        # Check if target labels are detected
        topk = torch.topk(probabilities, k=min(5, len(probabilities)))
        topk_indices = topk.indices
        topk_probabilities = topk.values
        
        # Get top-k label names
        detected_labels = []
        for idx, prob in zip(topk_indices, topk_probabilities):
            label = model.config.id2label[idx.item()]
            detected_labels.append(label)
        
        # Check if any target label is in detection results
        for target_label in target_labels:
            # Fuzzy matching: check if target label is contained in detected labels, or vice versa
            for detected_label in detected_labels:
                if target_label.lower() in detected_label.lower() or detected_label.lower() in target_label.lower():
                    return True
        
        return False
    else:
        # Return top-5 prediction results
        topk = torch.topk(probabilities, k=5)
        topk_indices = topk.indices
        topk_probabilities = topk.values
        
        results = []
        for idx, prob in zip(topk_indices, topk_probabilities):
            label = model.config.id2label[idx.item()]
            results.append((label, prob.item()))
        
        return results


def detect_object_in_video(
    video_path: str,
    target_objects: List[str],
    processor,
    model,
    num_samples: int = 5
) -> bool:
    """
    Detect if video contains target objects
    If any frame detects target objects, the video is considered to contain them
    
    Args:
        video_path: Path to video file
        target_objects: List of target objects
        processor: Image processor
        model: Classification model
        num_samples: Number of frames to sample
    
    Returns:
        True if target objects are detected, False otherwise
    """
    frames = sample_video_frames(video_path, num_samples)
    
    if len(frames) == 0:
        return False
    
    # Iterate through all sampled frames, return True if any frame detects target objects
    for frame in frames:
        detected = classify_frame(processor, model, frame, target_objects)
        if detected:
            return True
    
    return False


def evaluate_object_detection(
    input_folder: str,
    output_folder: Optional[str] = None,
    target_objects: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    num_samples: int = 5
):
    """
    Evaluate object detection for all videos in folder
    
    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder, if None then create 'output' folder under input folder
        target_objects: List of target objects, e.g., ['cassette player', 'panda', 'snoopy']
        model_path: Model path, if None then use default resnet-50
        num_samples: Number of frames to sample per video
    """
    # Load model
    if model_path is None:
        model_path = "microsoft/resnet-50"
    
    print(f"Loading model from {model_path}...")
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = ResNetForImageClassification.from_pretrained(model_path)
    
    if target_objects is None:
        target_objects = ['cassette player']
    
    # Set output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, "output")
    os.makedirs(output_folder, exist_ok=True)
    
    # List to store results
    results = []
    
    # Iterate through video files in folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    
    for filename in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(input_folder, filename)
        
        # Detection logic: if any frame detects target objects, consider them contained
        detected = detect_object_in_video(video_path, target_objects, processor, model, num_samples)
        
        res = {
            "video_path": video_path,
            "target_objects": ", ".join(target_objects),
            "detected": detected
        }
        print(f"{filename}: {'Detected' if detected else 'Not detected'}")
        results.append(res)
    
    # Save results
    if results:
        import pandas as pd
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_folder, "results.csv"), index=False)
        
        # Calculate detection rate
        total_videos = len(results)
        detected_count = sum(1 for r in results if r['detected'])
        detection_rate = detected_count / total_videos if total_videos > 0 else 0
        
        summary = {
            "total_videos": total_videos,
            "detected_count": detected_count,
            "detection_rate": detection_rate,
            "target_objects": ", ".join(target_objects)
        }
        
        import json
        with open(os.path.join(output_folder, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nSummary:")
        print(f"Total videos: {total_videos}")
        print(f"Detected: {detected_count}")
        print(f"Detection rate: {detection_rate:.2%}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate object detection in videos")
    parser.add_argument("--input-folder", type=str, required=True, help="Input folder containing videos")
    parser.add_argument("--output-folder", type=str, default=None, help="Output folder for results")
    parser.add_argument("--target-objects", type=str, nargs="+", default=["cassette player"], 
                       help="Target objects to detect")
    parser.add_argument("--model-path", type=str, default=None, 
                       help="Path to ResNet model (default: microsoft/resnet-50)")
    parser.add_argument("--num-samples", type=int, default=5, 
                       help="Number of frames to sample from each video")
    
    args = parser.parse_args()
    
    evaluate_object_detection(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        target_objects=args.target_objects,
        model_path=args.model_path,
        num_samples=args.num_samples
    )
