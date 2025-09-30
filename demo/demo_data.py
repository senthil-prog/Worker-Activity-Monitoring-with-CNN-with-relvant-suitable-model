import numpy as np
import cv2
import os
import json
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime, timedelta
import argparse

class DemoDataGenerator:
    """
    Generate synthetic demo data for worker activity monitoring.
    """
    
    def __init__(self, output_dir: str = "demo_data"):
        self.output_dir = output_dir
        self.class_names = [
            'sitting', 'standing', 'walking', 'working', 
            'resting', 'moving_objects', 'using_tools'
        ]
        
        # Create output directory structure
        self.setup_directories()
        
        # Activity-specific parameters
        self.activity_params = {
            'sitting': {'color': (100, 150, 200), 'shape': 'rectangle', 'size': (80, 60)},
            'standing': {'color': (200, 100, 150), 'shape': 'rectangle', 'size': (40, 120)},
            'walking': {'color': (150, 200, 100), 'shape': 'rectangle', 'size': (60, 100)},
            'working': {'color': (200, 200, 100), 'shape': 'rectangle', 'size': (70, 80)},
            'resting': {'color': (100, 200, 200), 'shape': 'circle', 'size': (60, 60)},
            'moving_objects': {'color': (200, 150, 100), 'shape': 'rectangle', 'size': (90, 70)},
            'using_tools': {'color': (150, 100, 200), 'shape': 'rectangle', 'size': (80, 90)}
        }
    
    def setup_directories(self):
        """Create directory structure for demo data."""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            split_dir = os.path.join(self.output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            for class_name in self.class_names:
                class_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
        
        print(f"Created directory structure in {self.output_dir}")
    
    def generate_activity_image(self, activity: str, image_size: tuple = (224, 224)) -> np.ndarray:
        """Generate a synthetic image representing a specific activity."""
        # Create base image
        image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # Add background noise
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        # Get activity parameters
        params = self.activity_params[activity]
        color = params['color']
        shape = params['shape']
        size = params['size']
        
        # Add random background elements
        for _ in range(random.randint(3, 8)):
            x = random.randint(0, image_size[0] - 20)
            y = random.randint(0, image_size[1] - 20)
            w = random.randint(10, 30)
            h = random.randint(10, 30)
            bg_color = tuple(random.randint(50, 150) for _ in range(3))
            cv2.rectangle(image, (x, y), (x + w, y + h), bg_color, -1)
        
        # Add main activity representation
        center_x = image_size[0] // 2
        center_y = image_size[1] // 2
        
        # Add some randomness to position
        center_x += random.randint(-20, 20)
        center_y += random.randint(-20, 20)
        
        if shape == 'rectangle':
            x1 = center_x - size[0] // 2
            y1 = center_y - size[1] // 2
            x2 = center_x + size[0] // 2
            y2 = center_y + size[1] // 2
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, image_size[0]))
            y1 = max(0, min(y1, image_size[1]))
            x2 = max(0, min(x2, image_size[0]))
            y2 = max(0, min(y2, image_size[1]))
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            
            # Add border
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
        elif shape == 'circle':
            radius = min(size) // 2
            cv2.circle(image, (center_x, center_y), radius, color, -1)
            cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Add activity-specific elements
        self.add_activity_elements(image, activity, center_x, center_y)
        
        # Add text label
        self.add_text_label(image, activity)
        
        # Apply some image augmentation
        image = self.apply_augmentation(image)
        
        return image
    
    def add_activity_elements(self, image: np.ndarray, activity: str, center_x: int, center_y: int):
        """Add activity-specific visual elements."""
        if activity == 'sitting':
            # Add chair-like elements
            cv2.rectangle(image, (center_x - 30, center_y + 20), (center_x + 30, center_y + 40), (139, 69, 19), -1)
        
        elif activity == 'standing':
            # Add ground line
            cv2.line(image, (center_x - 40, center_y + 60), (center_x + 40, center_y + 60), (100, 100, 100), 3)
        
        elif activity == 'walking':
            # Add movement lines
            for i in range(3):
                x = center_x - 20 + i * 20
                cv2.line(image, (x, center_y + 30), (x + 10, center_y + 40), (255, 255, 255), 2)
        
        elif activity == 'working':
            # Add desk/table
            cv2.rectangle(image, (center_x - 50, center_y + 30), (center_x + 50, center_y + 50), (160, 82, 45), -1)
            # Add tools
            cv2.rectangle(image, (center_x - 20, center_y - 10), (center_x - 10, center_y + 10), (192, 192, 192), -1)
        
        elif activity == 'resting':
            # Add relaxation elements
            cv2.circle(image, (center_x - 30, center_y - 20), 5, (255, 255, 0), -1)
            cv2.circle(image, (center_x + 30, center_y - 20), 5, (255, 255, 0), -1)
        
        elif activity == 'moving_objects':
            # Add objects being moved
            cv2.rectangle(image, (center_x - 40, center_y - 20), (center_x - 20, center_y), (255, 165, 0), -1)
            cv2.rectangle(image, (center_x + 20, center_y - 20), (center_x + 40, center_y), (255, 165, 0), -1)
        
        elif activity == 'using_tools':
            # Add tool representations
            cv2.line(image, (center_x - 30, center_y - 30), (center_x - 20, center_y - 20), (192, 192, 192), 3)
            cv2.line(image, (center_x + 20, center_y - 30), (center_x + 30, center_y - 20), (192, 192, 192), 3)
    
    def add_text_label(self, image: np.ndarray, activity: str):
        """Add text label to image."""
        try:
            # Convert to PIL for text rendering
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            
            # Try to use a default font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Add text
            text = activity.upper()
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text at top of image
            x = (image.shape[1] - text_width) // 2
            y = 10
            
            # Add background rectangle
            draw.rectangle([x - 5, y - 5, x + text_width + 5, y + text_height + 5], 
                          fill=(0, 0, 0), outline=(255, 255, 255))
            
            # Add text
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            # Convert back to numpy array
            image[:] = np.array(pil_image)
            
        except Exception as e:
            # Fallback: simple text using OpenCV
            cv2.putText(image, activity.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentation to image."""
        # Random brightness adjustment
        brightness = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Random contrast adjustment
        contrast = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        
        # Random noise
        if random.random() < 0.3:
            noise = np.random.randint(0, 20, image.shape, dtype=np.uint8)
            image = cv2.add(image, noise)
        
        # Random blur
        if random.random() < 0.2:
            kernel_size = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image
    
    def generate_dataset(self, samples_per_class: int = 100, splits: dict = None):
        """Generate complete dataset with train/val/test splits."""
        if splits is None:
            splits = {'train': 0.7, 'val': 0.2, 'test': 0.1}
        
        print(f"Generating dataset with {samples_per_class} samples per class...")
        
        total_samples = 0
        dataset_info = {
            'generation_timestamp': datetime.now().isoformat(),
            'samples_per_class': samples_per_class,
            'splits': splits,
            'class_names': self.class_names,
            'total_samples': 0,
            'split_distribution': {}
        }
        
        for class_name in self.class_names:
            print(f"Generating {class_name} images...")
            
            # Calculate samples per split
            train_samples = int(samples_per_class * splits['train'])
            val_samples = int(samples_per_class * splits['val'])
            test_samples = samples_per_class - train_samples - val_samples
            
            split_counts = {'train': train_samples, 'val': val_samples, 'test': test_samples}
            dataset_info['split_distribution'][class_name] = split_counts
            
            # Generate images for each split
            for split, count in split_counts.items():
                split_dir = os.path.join(self.output_dir, split, class_name)
                
                for i in range(count):
                    # Generate image
                    image = self.generate_activity_image(class_name)
                    
                    # Save image
                    filename = f"{class_name}_{split}_{i:04d}.jpg"
                    filepath = os.path.join(split_dir, filename)
                    cv2.imwrite(filepath, image)
                    
                    total_samples += 1
        
        dataset_info['total_samples'] = total_samples
        
        # Save dataset info
        info_path = os.path.join(self.output_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset generation completed!")
        print(f"Total samples: {total_samples}")
        print(f"Train samples: {int(total_samples * splits['train'])}")
        print(f"Validation samples: {int(total_samples * splits['val'])}")
        print(f"Test samples: {int(total_samples * splits['test'])}")
        print(f"Dataset info saved to: {info_path}")
        
        return dataset_info
    
    def create_sample_images(self, num_samples: int = 5):
        """Create sample images for each activity class."""
        print("Creating sample images...")
        
        sample_dir = os.path.join(self.output_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        for class_name in self.class_names:
            for i in range(num_samples):
                image = self.generate_activity_image(class_name)
                filename = f"sample_{class_name}_{i:02d}.jpg"
                filepath = os.path.join(sample_dir, filename)
                cv2.imwrite(filepath, image)
        
        print(f"Sample images saved to {sample_dir}")

def main():
    """Main function for generating demo data."""
    parser = argparse.ArgumentParser(description='Generate Demo Data for Worker Activity Monitoring')
    parser.add_argument('--output_dir', type=str, default='demo_data', help='Output directory')
    parser.add_argument('--samples_per_class', type=int, default=100, help='Samples per class')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--create_samples', action='store_true', help='Create sample images')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total_ratio}, not 1.0. Normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    splits = {
        'train': args.train_ratio,
        'val': args.val_ratio,
        'test': args.test_ratio
    }
    
    # Create generator
    generator = DemoDataGenerator(args.output_dir)
    
    # Generate dataset
    dataset_info = generator.generate_dataset(args.samples_per_class, splits)
    
    # Create sample images if requested
    if args.create_samples:
        generator.create_sample_images()
    
    print("Demo data generation completed successfully!")

if __name__ == "__main__":
    main()
