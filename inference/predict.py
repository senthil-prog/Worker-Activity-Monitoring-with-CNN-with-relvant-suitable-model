import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
import argparse
import json
from datetime import datetime
import sys
sys.path.append('..')

from models.cnn_model import create_model
from models.model_utils import load_model
from visualization.graphs import ActivityVisualizer

class BatchActivityPredictor:
    """
    Batch prediction system for worker activity monitoring.
    """
    
    def __init__(self, model_path: str, config: dict = None):
        self.model_path = model_path
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Activity classes
        self.class_names = [
            'sitting', 'standing', 'walking', 'working', 
            'resting', 'moving_objects', 'using_tools'
        ]
        
        # Load model
        self.model = self.load_model()
        self.model.eval()
        
        # Initialize visualizer
        self.visualizer = ActivityVisualizer(self.class_names)
        
        print(f"Batch predictor initialized on {self.device}")
        print(f"Model loaded from: {model_path}")
    
    def load_model(self):
        """Load the trained model."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model_type = checkpoint.get('model_type', 'standard')
                num_classes = checkpoint.get('num_classes', len(self.class_names))
                model = create_model(model_type, num_classes)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model = create_model('standard', len(self.class_names))
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image for prediction."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Convert to tensor and add batch dimension
            tensor = transform(image).unsqueeze(0)
            
            return tensor.to(self.device)
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def preprocess_batch(self, image_paths: list) -> torch.Tensor:
        """Preprocess batch of images for prediction."""
        batch_tensors = []
        valid_paths = []
        
        for image_path in image_paths:
            tensor = self.preprocess_image(image_path)
            if tensor is not None:
                batch_tensors.append(tensor)
                valid_paths.append(image_path)
        
        if not batch_tensors:
            return None, []
        
        # Stack tensors into batch
        batch = torch.cat(batch_tensors, dim=0)
        return batch, valid_paths
    
    def predict_single(self, image_path: str) -> dict:
        """Predict activity for a single image."""
        tensor = self.preprocess_image(image_path)
        if tensor is None:
            return None
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get prediction and confidence
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            # Get all class probabilities
            class_probabilities = probabilities.cpu().numpy()[0]
            
            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'predicted_activity': self.class_names[predicted_class],
                'confidence': confidence_score,
                'class_probabilities': {
                    self.class_names[i]: float(prob) for i, prob in enumerate(class_probabilities)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
    
    def predict_batch(self, image_paths: list, batch_size: int = 32) -> list:
        """Predict activities for a batch of images."""
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensor, valid_paths = self.preprocess_batch(batch_paths)
            
            if batch_tensor is None:
                continue
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get predictions and confidences
                confidences, predictions = torch.max(probabilities, 1)
                
                # Process each prediction in the batch
                for j, (image_path, pred, conf, probs) in enumerate(
                    zip(valid_paths, predictions, confidences, probabilities)
                ):
                    result = {
                        'image_path': image_path,
                        'predicted_class': pred.item(),
                        'predicted_activity': self.class_names[pred.item()],
                        'confidence': conf.item(),
                        'class_probabilities': {
                            self.class_names[k]: float(prob) for k, prob in enumerate(probs.cpu().numpy())
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
        
        return results
    
    def predict_directory(self, directory_path: str, extensions: list = None) -> list:
        """Predict activities for all images in a directory."""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all image files
        image_paths = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        if not image_paths:
            print("No images found in directory")
            return []
        
        # Predict activities
        results = self.predict_batch(image_paths)
        
        return results
    
    def save_predictions(self, results: list, output_path: str):
        """Save prediction results to JSON file."""
        output_data = {
            'predictions': results,
            'total_images': len(results),
            'class_names': self.class_names,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Predictions saved to {output_path}")
    
    def create_prediction_visualizations(self, results: list, output_dir: str):
        """Create visualizations for prediction results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract predictions and confidences
        predictions = [r['predicted_class'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Activity distribution
        self.visualizer.plot_activity_distribution(
            predictions,
            save_path=os.path.join(output_dir, 'activity_distribution.html')
        )
        
        # Confidence distribution
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Per-class confidence
        class_confidences = {name: [] for name in self.class_names}
        for result in results:
            class_confidences[result['predicted_activity']].append(result['confidence'])
        
        plt.figure(figsize=(12, 6))
        class_names = list(class_confidences.keys())
        avg_confidences = [np.mean(class_confidences[name]) if class_confidences[name] else 0 
                          for name in class_names]
        
        bars = plt.bar(class_names, avg_confidences, color='lightcoral', alpha=0.7)
        plt.title('Average Confidence by Activity Class')
        plt.xlabel('Activity Class')
        plt.ylabel('Average Confidence')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, conf in zip(bars, avg_confidences):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{conf:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_confidence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create comprehensive dashboard
        dashboard_path = self.visualizer.create_dashboard(
            predictions, save_path=os.path.join(output_dir, 'prediction_dashboard.html')
        )
        
        return dashboard_path
    
    def analyze_predictions(self, results: list) -> dict:
        """Analyze prediction results and generate statistics."""
        if not results:
            return {}
        
        predictions = [r['predicted_class'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Basic statistics
        analysis = {
            'total_predictions': len(results),
            'average_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'activity_distribution': {},
            'high_confidence_predictions': 0,
            'low_confidence_predictions': 0
        }
        
        # Activity distribution
        unique, counts = np.unique(predictions, return_counts=True)
        for i, count in zip(unique, counts):
            analysis['activity_distribution'][self.class_names[i]] = {
                'count': int(count),
                'percentage': float(count / len(results) * 100)
            }
        
        # Confidence analysis
        high_conf_threshold = 0.8
        low_conf_threshold = 0.5
        
        analysis['high_confidence_predictions'] = int(np.sum(np.array(confidences) > high_conf_threshold))
        analysis['low_confidence_predictions'] = int(np.sum(np.array(confidences) < low_conf_threshold))
        
        # Most/least confident predictions
        max_conf_idx = np.argmax(confidences)
        min_conf_idx = np.argmin(confidences)
        
        analysis['most_confident_prediction'] = {
            'image_path': results[max_conf_idx]['image_path'],
            'activity': results[max_conf_idx]['predicted_activity'],
            'confidence': results[max_conf_idx]['confidence']
        }
        
        analysis['least_confident_prediction'] = {
            'image_path': results[min_conf_idx]['image_path'],
            'activity': results[min_conf_idx]['predicted_activity'],
            'confidence': results[min_conf_idx]['confidence']
        }
        
        return analysis

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Batch Worker Activity Prediction')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory path')
    parser.add_argument('--output', type=str, default='predictions.json', help='Output JSON file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--output_dir', type=str, default='prediction_output', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = BatchActivityPredictor(args.model_path)
    
    # Determine input type
    if os.path.isfile(args.input):
        # Single image
        print(f"Predicting activity for single image: {args.input}")
        result = predictor.predict_single(args.input)
        if result:
            results = [result]
        else:
            print("Failed to process image")
            return
    elif os.path.isdir(args.input):
        # Directory
        print(f"Predicting activities for images in directory: {args.input}")
        results = predictor.predict_directory(args.input)
    else:
        print(f"Invalid input path: {args.input}")
        return
    
    if not results:
        print("No results to save")
        return
    
    # Save predictions
    predictor.save_predictions(results, args.output)
    
    # Generate analysis
    analysis = predictor.analyze_predictions(results)
    analysis_path = args.output.replace('.json', '_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis saved to {analysis_path}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total images processed: {analysis['total_predictions']}")
    print(f"Average confidence: {analysis['average_confidence']:.3f}")
    print(f"Most common activity: {max(analysis['activity_distribution'].items(), key=lambda x: x[1]['count'])[0]}")
    
    # Generate visualizations if requested
    if args.visualize:
        print("Generating visualizations...")
        dashboard_path = predictor.create_prediction_visualizations(results, args.output_dir)
        print(f"Visualizations saved to {args.output_dir}")
        print(f"Dashboard available at: {dashboard_path}")

if __name__ == "__main__":
    main()
