import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
import argparse
from datetime import datetime
import sys
sys.path.append('..')

from models.cnn_model import create_model
from models.model_utils import ModelTrainer, ModelEvaluator, save_model
from data.preprocessing import create_data_loaders
from visualization.graphs import ActivityVisualizer

class TrainingManager:
    """
    Comprehensive training manager for worker activity monitoring CNN.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = [
            'sitting', 'standing', 'walking', 'working', 
            'resting', 'moving_objects', 'using_tools'
        ]
        
        # Create output directory
        self.output_dir = config.get('output_dir', 'training_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = ActivityVisualizer(self.class_names)
        
        print(f"Training on device: {self.device}")
        print(f"Output directory: {self.output_dir}")
    
    def setup_data(self):
        """Setup data loaders."""
        print("Setting up data loaders...")
        
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 4),
            image_size=self.config.get('image_size', (224, 224))
        )
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
    
    def setup_model(self):
        """Setup model and training components."""
        print("Setting up model...")
        
        # Create model
        self.model = create_model(
            model_type=self.config.get('model_type', 'standard'),
            num_classes=len(self.class_names),
            pretrained=self.config.get('pretrained', False)
        )
        self.model = self.model.to(self.device)
        
        # Initialize trainer and evaluator
        self.trainer = ModelTrainer(self.model, self.device, len(self.class_names))
        self.evaluator = ModelEvaluator(self.model, self.device, self.class_names)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self):
        """Train the model."""
        print("Starting training...")
        
        # Train the model
        train_losses, val_losses, train_accs, val_accs = self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.config.get('epochs', 50),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4),
            patience=self.config.get('patience', 10)
        )
        
        # Save training history
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accs,
            'val_accuracies': val_accs,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
        
        return training_history
    
    def evaluate(self):
        """Evaluate the trained model."""
        print("Evaluating model...")
        
        # Evaluate on test set
        predictions, targets, probabilities = self.evaluator.evaluate(self.test_loader)
        
        # Generate evaluation metrics
        classification_report = self.evaluator.generate_classification_report(predictions, targets)
        
        # Save evaluation results
        evaluation_results = {
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'probabilities': probabilities.tolist(),
            'classification_report': classification_report,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print("Classification Report:")
        print(classification_report)
        
        return evaluation_results
    
    def create_visualizations(self, training_history: dict, evaluation_results: dict):
        """Create comprehensive visualizations."""
        print("Creating visualizations...")
        
        # Training curves
        self.evaluator.plot_training_history(
            training_history['train_losses'],
            training_history['val_losses'],
            training_history['train_accuracies'],
            training_history['val_accuracies'],
            save_path=os.path.join(self.output_dir, 'training_curves.png')
        )
        
        # Confusion matrix
        self.evaluator.plot_confusion_matrix(
            evaluation_results['targets'],
            evaluation_results['predictions'],
            save_path=os.path.join(self.output_dir, 'confusion_matrix.png')
        )
        
        # Activity distribution
        self.visualizer.plot_activity_distribution(
            evaluation_results['predictions'],
            save_path=os.path.join(self.output_dir, 'activity_distribution.html')
        )
        
        # Activity timeline
        self.visualizer.plot_activity_timeline(
            evaluation_results['predictions'],
            save_path=os.path.join(self.output_dir, 'activity_timeline.html')
        )
        
        # Activity statistics
        self.visualizer.plot_activity_statistics(
            evaluation_results['predictions'],
            save_path=os.path.join(self.output_dir, 'activity_statistics.html')
        )
        
        # Comprehensive dashboard
        dashboard_path = self.visualizer.create_dashboard(
            evaluation_results['predictions'],
            evaluation_results['targets'],
            training_history['train_losses'],
            training_history['val_losses'],
            training_history['train_accuracies'],
            training_history['val_accuracies'],
            save_path=os.path.join(self.output_dir, 'training_dashboard.html')
        )
        
        print(f"All visualizations saved to {self.output_dir}")
        return dashboard_path
    
    def save_model_checkpoint(self, training_history: dict, evaluation_results: dict):
        """Save model checkpoint with metadata."""
        checkpoint_path = os.path.join(self.output_dir, 'model_checkpoint.pth')
        
        additional_info = {
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'config': self.config,
            'class_names': self.class_names,
            'timestamp': datetime.now().isoformat()
        }
        
        save_model(self.model, checkpoint_path, additional_info)
        print(f"Model checkpoint saved to {checkpoint_path}")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        print("=" * 60)
        print("WORKER ACTIVITY MONITORING - CNN TRAINING")
        print("=" * 60)
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Training
        training_history = self.train()
        
        # Evaluation
        evaluation_results = self.evaluate()
        
        # Visualizations
        dashboard_path = self.create_visualizations(training_history, evaluation_results)
        
        # Save model
        self.save_model_checkpoint(training_history, evaluation_results)
        
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {self.output_dir}")
        print(f"Dashboard available at: {dashboard_path}")
        
        return {
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'dashboard_path': dashboard_path,
            'output_dir': self.output_dir
        }

def create_default_config():
    """Create default training configuration."""
    return {
        'data_dir': 'dummy_data',
        'output_dir': 'training_output',
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'patience': 10,
        'model_type': 'standard',
        'pretrained': False,
        'image_size': (224, 224),
        'num_workers': 4
    }

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Worker Activity Monitoring CNN')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='dummy_data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='training_output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_type', type=str, default='standard', 
                       choices=['standard', 'lightweight'], help='Model type')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    config.update({
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'model_type': args.model_type,
        'pretrained': args.pretrained
    })
    
    # Create and run training manager
    trainer = TrainingManager(config)
    results = trainer.run_training_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()
