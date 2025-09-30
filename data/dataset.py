import torch
from torch.utils.data import Dataset
import os
import json
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import cv2

class WorkerActivityDataset(Dataset):
    """
    Enhanced dataset class for worker activity monitoring with additional features.
    """
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 transform=None, image_size: Tuple[int, int] = (224, 224),
                 use_cache: bool = True):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.use_cache = use_cache
        
        # Activity classes with descriptions
        self.class_info = {
            'sitting': {'id': 0, 'description': 'Worker sitting at workstation'},
            'standing': {'id': 1, 'description': 'Worker standing upright'},
            'walking': {'id': 2, 'description': 'Worker walking or moving around'},
            'working': {'id': 3, 'description': 'Worker actively working at task'},
            'resting': {'id': 4, 'description': 'Worker taking a break or resting'},
            'moving_objects': {'id': 5, 'description': 'Worker moving or carrying objects'},
            'using_tools': {'id': 6, 'description': 'Worker using tools or equipment'}
        }
        
        self.class_names = list(self.class_info.keys())
        self.class_to_idx = {name: info['id'] for name, info in self.class_info.items()}
        
        # Load metadata
        self.metadata_file = os.path.join(data_dir, f'{split}_metadata.json')
        self.samples = self._load_samples()
        
        # Cache for loaded images
        self.image_cache = {} if use_cache else None
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_class_distribution()
    
    def _load_samples(self) -> List[Dict]:
        """Load samples with metadata."""
        samples = []
        
        # Try to load from metadata file first
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                samples = metadata.get('samples', [])
        else:
            # Load from directory structure
            samples = self._load_from_directory()
            # Save metadata for future use
            self._save_metadata(samples)
        
        return samples
    
    def _load_from_directory(self) -> List[Dict]:
        """Load samples from directory structure."""
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
                    
                    # Get image metadata
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                    except:
                        width, height = self.image_size
                    
                    sample = {
                        'image_path': image_path,
                        'label': class_idx,
                        'class_name': class_name,
                        'filename': filename,
                        'width': width,
                        'height': height,
                        'split': self.split
                    }
                    samples.append(sample)
        
        return samples
    
    def _create_dummy_samples(self) -> List[Dict]:
        """Create dummy samples for demonstration."""
        dummy_samples = []
        for class_idx, class_name in enumerate(self.class_names):
            for i in range(10):  # 10 samples per class
                sample = {
                    'image_path': f"dummy_{class_name}_{i}.jpg",
                    'label': class_idx,
                    'class_name': class_name,
                    'filename': f"dummy_{class_name}_{i}.jpg",
                    'width': self.image_size[0],
                    'height': self.image_size[1],
                    'split': self.split
                }
                dummy_samples.append(sample)
        return dummy_samples
    
    def _save_metadata(self, samples: List[Dict]):
        """Save dataset metadata to JSON file."""
        metadata = {
            'samples': samples,
            'class_info': self.class_info,
            'total_samples': len(samples),
            'split': self.split
        }
        
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _print_class_distribution(self):
        """Print class distribution statistics."""
        distribution = {name: 0 for name in self.class_names}
        for sample in self.samples:
            distribution[sample['class_name']] += 1
        
        print("Class Distribution:")
        for class_name, count in distribution.items():
            percentage = (count / len(self.samples)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        sample = self.samples[idx]
        
        # Load image
        if self.use_cache and sample['image_path'] in self.image_cache:
            image = self.image_cache[sample['image_path']]
        else:
            image = self._load_image(sample['image_path'])
            if self.use_cache:
                self.image_cache[sample['image_path']] = image
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return image, label, and metadata
        return image, sample['label'], sample
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        if image_path.startswith('dummy_'):
            # Create dummy image
            return self._create_dummy_image()
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self._create_dummy_image()
    
    def _create_dummy_image(self) -> Image.Image:
        """Create a dummy image for demonstration."""
        # Create a random RGB image with some structure
        dummy_array = np.random.randint(0, 255, (self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
        cv2.circle(dummy_array, (center_x, center_y), 50, (100, 150, 200), -1)
        
        return Image.fromarray(dummy_array)
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data."""
        class_counts = {name: 0 for name in self.class_names}
        for sample in self.samples:
            class_counts[sample['class_name']] += 1
        
        total_samples = len(self.samples)
        num_classes = len(self.class_names)
        
        weights = []
        for class_name in self.class_names:
            if class_counts[class_name] > 0:
                weight = total_samples / (num_classes * class_counts[class_name])
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sample_by_class(self, class_name: str, num_samples: int = 1) -> List[Dict]:
        """Get samples from a specific class."""
        class_samples = [s for s in self.samples if s['class_name'] == class_name]
        return class_samples[:num_samples]
    
    def visualize_samples(self, num_samples_per_class: int = 2, save_path: Optional[str] = None):
        """Visualize sample images from each class."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(self.class_names), num_samples_per_class, 
                                figsize=(num_samples_per_class * 3, len(self.class_names) * 3))
        
        if len(self.class_names) == 1:
            axes = axes.reshape(1, -1)
        if num_samples_per_class == 1:
            axes = axes.reshape(-1, 1)
        
        for class_idx, class_name in enumerate(self.class_names):
            class_samples = self.get_sample_by_class(class_name, num_samples_per_class)
            
            for sample_idx in range(num_samples_per_class):
                if sample_idx < len(class_samples):
                    sample = class_samples[sample_idx]
                    image = self._load_image(sample['image_path'])
                    
                    axes[class_idx, sample_idx].imshow(image)
                    axes[class_idx, sample_idx].set_title(f"{class_name}\n{sample['filename']}")
                    axes[class_idx, sample_idx].axis('off')
                else:
                    axes[class_idx, sample_idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ActivitySequenceDataset(Dataset):
    """
    Dataset for sequence-based activity recognition (for future extension).
    """
    
    def __init__(self, data_dir: str, sequence_length: int = 16, 
                 transform=None, image_size: Tuple[int, int] = (224, 224)):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.image_size = image_size
        
        # Load video sequences
        self.sequences = self._load_sequences()
    
    def _load_sequences(self) -> List[Dict]:
        """Load video sequences with activity labels."""
        # This would load video sequences from the data directory
        # For now, return empty list
        return []
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sequence_info = self.sequences[idx]
        
        # Load sequence of frames
        frames = []
        for frame_path in sequence_info['frame_paths']:
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        
        # Stack frames into sequence tensor
        sequence = torch.stack(frames)
        
        return sequence, sequence_info['label']

if __name__ == "__main__":
    # Test the enhanced dataset
    print("Testing enhanced dataset...")
    
    # Create dummy data directory
    dummy_data_dir = "dummy_data"
    os.makedirs(dummy_data_dir, exist_ok=True)
    
    # Test dataset creation
    dataset = WorkerActivityDataset(dummy_data_dir, 'train')
    print(f"Dataset length: {len(dataset)}")
    
    # Test class weights
    weights = dataset.get_class_weights()
    print(f"Class weights: {weights}")
    
    # Test sample retrieval
    sample = dataset[0]
    print(f"Sample shape: {sample[0].shape}, Label: {sample[1]}, Class: {sample[2]['class_name']}")
    
    print("Enhanced dataset test completed!")
