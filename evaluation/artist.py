import base64
import os
import cv2
import numpy as np
import pandas as pd
from openai import OpenAI
import httpx
from typing import Optional, List


def encode_image_from_frame(frame):
    """Encode video frame to base64 string"""
    ret, buffer = cv2.imencode('.jpg', frame)
    if ret:
        return base64.b64encode(buffer).decode('utf-8')
    return None


def sample_video_frames(video_path: str, num_samples: int = 5) -> List[np.ndarray]:
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


def get_predicted_artist_from_frame(client: OpenAI, frame: np.ndarray, model: str = "gpt-4o-mini") -> Optional[str]:
    """
    Predict artistic style from a single frame
    
    Args:
        client: OpenAI client
        frame: Video frame
        model: Model name to use
    
    Returns:
        Predicted artist name, or None if failed
    """
    prompt = '''Given an input image of artwork, classify it among the following five artists by their style and return only the index number of the most likely artist. 
    The artists are:
    1 'Tyler Edlin'
    2 'Thomas Kinkade'
    3 'Kilian Eng'
    4 'Kelly Mckernan'
    5 'Ajin: Demi-Human'
    Ensure output only the number corresponding to the most likely artist.'''
    
    base64_image = encode_image_from_frame(frame)
    if base64_image is None:
        return None
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=1
        )
        
        if response.choices:
            response_content = response.choices[0].message.content.strip()
            if response_content.isdigit():
                predicted_index = int(response_content)
                if 1 <= predicted_index <= 5:
                    artist_names = ['Tyler Edlin', 'Thomas Kinkade', 'Kilian Eng', 'Kelly Mckernan', 'Ajin: Demi-Human']
                    return artist_names[predicted_index - 1]
    except Exception as e:
        print(f"Error predicting artist from frame: {e}")
    
    return None


def detect_artist_in_video(video_path: str, target_artist: str, client: OpenAI, num_samples: int = 5) -> bool:
    """
    Detect if video contains target artistic style
    If any frame detects the target artistic style, the video is considered to contain it
    
    Args:
        video_path: Path to video file
        target_artist: Target artist name
        client: OpenAI client
        num_samples: Number of frames to sample
    
    Returns:
        True if target artistic style is detected, False otherwise
    """
    frames = sample_video_frames(video_path, num_samples)
    
    if len(frames) == 0:
        return False
    
    # Iterate through all sampled frames, return True if any frame detects target artistic style
    for frame in frames:
        predicted_artist = get_predicted_artist_from_frame(client, frame)
        if predicted_artist == target_artist:
            return True
    
    return False


def evaluate_artistic_style(
    input_folder: str,
    output_folder: Optional[str] = None,
    expected_artists: Optional[dict] = None,
    num_samples: int = 5,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
):
    """
    Evaluate artistic style detection for all videos in folder
    
    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder, if None then create 'output' folder under input folder
        expected_artists: Expected artists dictionary, format: {video index range: artist name}
        num_samples: Number of frames to sample per video
        api_key: OpenAI API key, if None then read from environment variable
        base_url: OpenAI API base URL, if None then use default value
    """
    # Initialize OpenAI client
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(
            base_url=base_url,
            follow_redirects=True,
        ),
    )
    
    # Define artist ranges (if not provided)
    if expected_artists is None:
        expected_artists = {
            'Tyler Edlin': range(0, 40),
            'Thomas Kinkade': range(40, 80),
            'Kilian Eng': range(80, 120),
            'Kelly Mckernan': range(120, 160),
            'Ajin: Demi-Human': range(160, 200)
        }
    
    # Set output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, "output")
    os.makedirs(output_folder, exist_ok=True)
    
    # List to store results
    results = []
    
    # Iterate through video files in folder
    for filename in os.listdir(input_folder):
        if not filename.endswith(".mp4"):
            continue
        
        # Extract index and artist
        try:
            index = int(filename.split('-')[0])
        except ValueError:
            continue
        
        original_artist = None
        for artist, idx_range in expected_artists.items():
            if index in idx_range:
                original_artist = artist
                break
        
        if original_artist is None:
            continue
        
        video_path = os.path.join(input_folder, filename)
        
        # Detection logic: if any frame detects target artistic style, consider it contained
        detected = detect_artist_in_video(video_path, original_artist, client, num_samples)
        accuracy = 1 if detected else 0
        
        res = {
            "video_path": video_path,
            "original_artist": original_artist,
            "detected": detected,
            "accuracy": accuracy
        }
        print(res)
        results.append(res)
    
    # Create DataFrame and save to CSV file
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_folder, "results.csv"), index=False)
        
        # Calculate prediction accuracy for each artist category
        accuracy_summary = results_df.groupby('original_artist')['accuracy'].mean().reset_index()
        accuracy_summary.columns = ['original_artist', 'average_accuracy']
        accuracy_summary.to_csv(os.path.join(output_folder, "accuracy_summary.csv"), index=False)
    else:
        print("No results to save.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate artistic style detection in videos")
    parser.add_argument("--input-folder", type=str, required=True, help="Input folder containing videos")
    parser.add_argument("--output-folder", type=str, default=None, help="Output folder for results")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of frames to sample from each video")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI API base URL (or set OPENAI_BASE_URL env var)")
    
    args = parser.parse_args()
    
    evaluate_artistic_style(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        num_samples=args.num_samples,
        api_key=args.api_key,
        base_url=args.base_url
    )
