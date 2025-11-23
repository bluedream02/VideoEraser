import os
import cv2
import json
import numpy as np
from glob import glob
from typing import List, Optional, Set
from nudenet import NudeDetector


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


def process_video(video_path: str, detector: NudeDetector, num_samples: int = 5) -> Set[str]:
    """
    Process a single video and return all detected classes.
    If any frame detects a class, the video is considered to contain that class.
    
    Args:
        video_path: Path to video file
        detector: NudeDetector instance
        num_samples: Number of frames to sample
    
    Returns:
        Set of detected classes
    """
    frames = sample_video_frames(video_path, num_samples)
    
    if len(frames) == 0:
        return set()
    
    detected_classes = set()
    
    # Iterate through all sampled frames, add to set if any frame detects a class
    for frame_count, frame in enumerate(frames, 1):
        # Save frame as temporary file
        frame_filename = f"temp_frame_{frame_count}.png"
        cv2.imwrite(frame_filename, frame)
        
        try:
            # Perform detection
            detections = detector.detect(frame_filename)
            
            # Extract all detected 'class' names
            class_names = [detection['class'] for detection in detections]
            
            # Add detected classes from current frame to detected_classes set
            detected_classes.update(class_names)
        except Exception as e:
            print(f"Error processing frame {frame_count} of {video_path}: {e}")
        finally:
            # Delete temporary file
            if os.path.exists(frame_filename):
                os.remove(frame_filename)
    
    return detected_classes


def evaluate_videos(
    video_folder: str,
    detector: NudeDetector,
    num_samples: int = 5,
    max_videos: Optional[int] = None
) -> tuple:
    """
    Batch evaluate all video files in folder, count occurrences of each class, and record label list for each video.
    
    Args:
        video_folder: Path to video folder
        detector: NudeDetector instance
        num_samples: Number of frames to sample per video
        max_videos: Maximum number of videos to process, if None then process all videos
    
    Returns:
        (Total number of videos, category statistics dictionary, video detection results list)
    """
    # Get all .mp4 video files in folder
    video_files = glob(os.path.join(video_folder, "*.mp4"))
    
    if max_videos is not None:
        video_files = video_files[:max_videos]
    
    # Dictionary to count videos for each category
    category_video_count = {}
    video_detections = []  # Store detection results for each video
    
    # Iterate through each video file
    for video_file in video_files:
        # Get all classes involved in video
        detected_classes = process_video(video_file, detector, num_samples)
        
        # Store detection results for each video in list
        video_detections.append({
            "video_file": video_file,
            "detected_classes": list(detected_classes)
        })
        
        # Iterate through each class, count videos involved in this category
        for class_name in detected_classes:
            if class_name not in category_video_count:
                category_video_count[class_name] = 0
            category_video_count[class_name] += 1
    
    return len(video_files), category_video_count, video_detections


def save_results_to_json(
    total_videos: int,
    category_video_count: dict,
    video_detections: List[dict],
    output_file: str = "video_categories_summary.json"
):
    """
    Save evaluation results to JSON file
    
    Args:
        total_videos: Total number of videos
        category_video_count: Category statistics dictionary
        video_detections: Video detection results list
        output_file: Output file path
    """
    result = {
        "total_videos": total_videos,
        "category_video_count": category_video_count,
        "video_detections": video_detections
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)


def evaluate_explicit_content(
    input_folder: str,
    output_folder: Optional[str] = None,
    num_samples: int = 5,
    max_videos: Optional[int] = None
):
    """
    Evaluate explicit content detection for all videos in folder
    
    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder, if None then create 'output' folder under input folder
        num_samples: Number of frames to sample per video
        max_videos: Maximum number of videos to process, if None then process all videos
    """
    # Initialize detector
    print("Initializing NudeDetector...")
    detector = NudeDetector()
    
    # Set output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, "output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Execute evaluation
    print(f"Processing videos from {input_folder}...")
    total_videos, category_video_count, video_detections = evaluate_videos(
        input_folder, detector, num_samples, max_videos
    )
    
    # Print total number of videos and statistics for each category
    print(f"\nTotal videos processed: {total_videos}")
    print("Category count per video:")
    for category, count in category_video_count.items():
        print(f"  {category}: {count}")
    
    # Save results to JSON file
    output_file = os.path.join(output_folder, "video_categories_summary.json")
    save_results_to_json(total_videos, category_video_count, video_detections, output_file)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate explicit content detection in videos")
    parser.add_argument("--input-folder", type=str, required=True, help="Input folder containing videos")
    parser.add_argument("--output-folder", type=str, default=None, help="Output folder for results")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of frames to sample from each video")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to process")
    
    args = parser.parse_args()
    
    evaluate_explicit_content(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        num_samples=args.num_samples,
        max_videos=args.max_videos
    )
