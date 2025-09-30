import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import os
import json
from typing import List, Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WorkerActivityDataset(Dataset):
    """
    Custom dataset for worker activity classification.
    
    Expected directory structure:
    data/
    ├── train/
    │   ├── sitting/
    │   ├── standing/
    │   ├── walking/
    │   ├── working/
    │   ├── resting/
    │   ├── moving_objects/
    │   └── using_tools/
    ├── val/
    └── test/
    """
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 transform: Optional[transforms.Compose] = None,
                 image_size: Tuple[int, int] = (224, 224)):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Activity classes
        self.class_names = [
            'sitting', 'standing', 'walking', 'working', 
            'resting', 'moving_objects', 'using_tools'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load data
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and labels."""
        samples = []
        split_dir = os.path.join(self.data_dir, self.split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist. Creating dummy data...")
            return self._create_dummy_samples()
        
        for class_name in self.class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(class_dir, filename)
                    samples.append((image_path, class_idx))
        
        return samples
    
    def _create_dummy_samples(self) -> List[Tuple[str, int]]:
        """Create dummy samples for demonstration."""
        dummy_samples = []
        for class_idx in range(len(self.class_names)):
            # Create 10 dummy samples per class
            for i in range(10):
                dummy_path = f"dummy_{self.class_names[class_idx]}_{i}.jpg"
                dummy_samples.append((dummy_path, class_idx))
        return dummy_samples
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution in the dataset."""
        distribution = {name: 0 for name in self.class_names}
        for _, class_idx in self.samples:
            distribution[self.class_names[class_idx]] += 1
        return distribution
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        
        # Load image
        if image_path.startswith('dummy_'):
            # Create dummy image
            image = self._create_dummy_image()
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                image = self._create_dummy_image()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _create_dummy_image(self) -> Image.Image:
        """Create a dummy image for demonstration."""
        # Create a random RGB image
        dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(dummy_array)

class DataPreprocessor:
    """Data preprocessing utilities for worker activity monitoring."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        
    def get_train_transforms(self, use_augmentation: bool = True) -> transforms.Compose:
        """Get training transforms with data augmentation."""
        if use_augmentation:
            transform_list = [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            transform_list = [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        
        return transforms.Compose(transform_list)
    
    def get_val_transforms(self) -> transforms.Compose:
        """Get validation transforms."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_albumentations_transforms(self, is_training: bool = True) -> A.Compose:
        """Get Albumentations transforms for advanced augmentation."""
        if is_training:
            return A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(224, 224),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.GaussNoise(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

class VideoFrameExtractor:
    """Extract frames from video files for activity monitoring."""
    
    def __init__(self, output_size: Tuple[int, int] = (224, 224)):
        self.output_size = output_size
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      frame_interval: int = 30, max_frames: int = 1000) -> List[str]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame
            max_frames: Maximum number of frames to extract
        
        Returns:
            List of extracted frame paths
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        frame_count = 0
        extracted_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Resize frame
                frame = cv2.resize(frame, self.output_size)
                
                # Save frame
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)
                
                frame_idx += 1
                
                if frame_idx >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(extracted_frames)} frames from {video_path}")
        return extracted_frames
    
    def extract_frames_with_activity_labels(self, video_path: str, 
                                          activity_labels: List[str],
                                          output_dir: str) -> Dict[str, List[str]]:
        """
        Extract frames with activity labels for training data creation.
        
        Args:
            video_path: Path to input video
            activity_labels: List of activity labels for each time segment
            output_dir: Directory to save labeled frames
        
        Returns:
            Dictionary mapping activity labels to frame paths
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate frame ranges for each activity
        segment_duration = duration / len(activity_labels)
        frames_per_segment = total_frames // len(activity_labels)
        
        labeled_frames = {label: [] for label in set(activity_labels)}
        
        for segment_idx, activity_label in enumerate(activity_labels):
            start_frame = segment_idx * frames_per_segment
            end_frame = min((segment_idx + 1) * frames_per_segment, total_frames)
            
            # Create activity directory
            activity_dir = os.path.join(output_dir, activity_label)
            os.makedirs(activity_dir, exist_ok=True)
            
            # Extract frames for this activity
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_idx = 0
            while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize and save frame
                frame = cv2.resize(frame, self.output_size)
                frame_filename = f"{activity_label}_{frame_idx:06d}.jpg"
                frame_path = os.path.join(activity_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                labeled_frames[activity_label].append(frame_path)
                
                frame_idx += 1
        
        cap.release()
        return labeled_frames

def create_data_loaders(data_dir: str, batch_size: int = 32, 
                       num_workers: int = 4, image_size: Tuple[int, int] = (224, 224)) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Root directory containing train/val/test splits
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        image_size: Target image size
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    preprocessor = DataPreprocessor(image_size)
    
    # Create datasets
    train_dataset = WorkerActivityDataset(
        data_dir, 'train', 
        transform=preprocessor.get_train_transforms(use_augmentation=True),
        image_size=image_size
    )
    
    val_dataset = WorkerActivityDataset(
        data_dir, 'val',
        transform=preprocessor.get_val_transforms(),
        image_size=image_size
    )
    
    test_dataset = WorkerActivityDataset(
        data_dir, 'test',
        transform=preprocessor.get_val_transforms(),
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the data preprocessing
    print("Testing data preprocessing...")
    
    # Create dummy data directory structure
    dummy_data_dir = "dummy_data"
    os.makedirs(dummy_data_dir, exist_ok=True)
    
    # Test dataset creation
    dataset = WorkerActivityDataset(dummy_data_dir, 'train')
    print(f"Dataset length: {len(dataset)}")
    
    # Test data loaders
    train_loader, val_loader, test_loader = create_data_loaders(dummy_data_dir, batch_size=8)
    
    # Test batch loading
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        if batch_idx >= 2:  # Test only first few batches
            break
    
    print("Data preprocessing test completed!")
