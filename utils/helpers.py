"""
Utility functions for Worker Activity Monitoring system.
"""

import os
import json
import time
import random
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

def get_device() -> torch.device:
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    return device

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def create_timestamp() -> str:
    """Create timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(path: str) -> str:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)
    return path

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file."""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate accuracy."""
    return np.mean(predictions == targets)

def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets."""
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    weights = []
    for count in class_counts:
        if count > 0:
            weight = total_samples / (num_classes * count)
        else:
            weight = 1.0
        weights.append(weight)
    
    return torch.FloatTensor(weights)

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax1.plot(history['train_acc'], label='Training Accuracy', color='blue')
    ax1.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix_plot(y_true: List[int], y_pred: List[int], 
                               class_names: List[str], save_path: Optional[str] = None) -> None:
    """Create confusion matrix plot."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def resize_image_with_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int], 
                                 fill_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Resize image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create canvas with target size
    canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    
    # Calculate position to center the image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas

def create_video_from_images(image_paths: List[str], output_path: str, 
                           fps: int = 30, frame_size: Tuple[int, int] = (640, 480)) -> None:
    """Create video from sequence of images."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                # Resize image to frame size
                image = cv2.resize(image, frame_size)
                out.write(image)
    
    out.release()
    print(f"Video created: {output_path}")

def extract_frames_from_video(video_path: str, output_dir: str, 
                            frame_interval: int = 30) -> List[str]:
    """Extract frames from video file."""
    ensure_directory(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {len(extracted_frames)} frames from {video_path}")
    return extracted_frames

def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate model size in different units."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'parameters_mb': param_size / 1024**2,
        'buffers_mb': buffer_size / 1024**2,
        'total_mb': size_all_mb,
        'total_gb': size_all_mb / 1024
    }

def benchmark_model(model: torch.nn.Module, input_shape: Tuple[int, ...], 
                   num_iterations: int = 100, device: str = 'cpu') -> Dict[str, float]:
    """Benchmark model inference speed."""
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    return {
        'total_time': total_time,
        'avg_inference_time': avg_time,
        'fps': fps,
        'num_iterations': num_iterations
    }

def create_activity_summary(predictions: List[int], class_names: List[str], 
                          timestamps: Optional[List[datetime]] = None) -> Dict[str, Any]:
    """Create activity summary from predictions."""
    if timestamps is None:
        timestamps = [datetime.now() - timedelta(minutes=len(predictions)-i) 
                     for i in range(len(predictions))]
    
    # Calculate statistics
    unique, counts = np.unique(predictions, return_counts=True)
    activity_counts = {class_names[i]: count for i, count in zip(unique, counts)}
    
    # Calculate percentages
    total_predictions = len(predictions)
    activity_percentages = {name: (count / total_predictions) * 100 
                           for name, count in activity_counts.items()}
    
    # Find most common activity
    most_common = max(activity_counts.items(), key=lambda x: x[1])
    
    # Calculate time spans
    if len(timestamps) > 1:
        total_duration = (timestamps[-1] - timestamps[0]).total_seconds() / 60  # minutes
        avg_activity_duration = total_duration / len(unique)
    else:
        total_duration = 0
        avg_activity_duration = 0
    
    summary = {
        'total_predictions': total_predictions,
        'total_duration_minutes': total_duration,
        'activity_distribution': activity_counts,
        'activity_percentages': activity_percentages,
        'most_common_activity': most_common[0],
        'most_common_count': most_common[1],
        'unique_activities': len(unique),
        'avg_activity_duration_minutes': avg_activity_duration,
        'timestamp_range': {
            'start': timestamps[0].isoformat(),
            'end': timestamps[-1].isoformat()
        }
    }
    
    return summary

def validate_image_file(filepath: str) -> bool:
    """Validate if file is a valid image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_image_info(filepath: str) -> Dict[str, Any]:
    """Get image information."""
    try:
        with Image.open(filepath) as img:
            return {
                'filename': os.path.basename(filepath),
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
    except Exception as e:
        return {'error': str(e)}

def create_progress_bar(total: int, description: str = "Processing") -> None:
    """Create a simple progress bar."""
    from tqdm import tqdm
    return tqdm(total=total, desc=description, unit="item")

def log_performance_metrics(metrics: Dict[str, Any], log_file: str) -> None:
    """Log performance metrics to file."""
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'metrics': metrics
    }
    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def cleanup_old_files(directory: str, pattern: str, max_age_days: int = 7) -> int:
    """Clean up old files matching pattern."""
    import glob
    from pathlib import Path
    
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    files_to_clean = glob.glob(os.path.join(directory, pattern))
    cleaned_count = 0
    
    for file_path in files_to_clean:
        file_age = current_time - os.path.getmtime(file_path)
        if file_age > max_age_seconds:
            try:
                os.remove(file_path)
                cleaned_count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    print(f"Cleaned up {cleaned_count} old files from {directory}")
    return cleaned_count

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test random seed
    set_random_seed(42)
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test timestamp
    timestamp = create_timestamp()
    print(f"Timestamp: {timestamp}")
    
    # Test directory creation
    test_dir = ensure_directory("test_directory")
    print(f"Created directory: {test_dir}")
    
    # Test JSON operations
    test_data = {"test": "data", "number": 42}
    save_json(test_data, "test_data.json")
    loaded_data = load_json("test_data.json")
    print(f"JSON test: {loaded_data}")
    
    # Cleanup
    os.remove("test_data.json")
    os.rmdir("test_directory")
    
    print("Utility functions test completed!")
