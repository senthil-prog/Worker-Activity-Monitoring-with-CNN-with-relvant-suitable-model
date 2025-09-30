import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import argparse
from datetime import datetime
import sys
sys.path.append('..')

from models.cnn_model import create_model
from models.model_utils import ModelEvaluator, load_model
from data.preprocessing import create_data_loaders
from visualization.graphs import ActivityVisualizer

class ModelEvaluator:
    """
    Comprehensive model evaluation class for worker activity monitoring.
    """
    
    def __init__(self, model_path: str, config: dict = None):
        self.model_path = model_path
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = [
            'sitting', 'standing', 'walking', 'working', 
            'resting', 'moving_objects', 'using_tools'
        ]
        
        # Load model
        self.model = self.load_model()
        self.evaluator = ModelEvaluator(self.model, self.device, self.class_names)
        self.visualizer = ActivityVisualizer(self.class_names)
        
        print(f"Model loaded from: {model_path}")
        print(f"Device: {self.device}")
    
    def load_model(self):
        """Load the trained model."""
        try:
            # Try to load as checkpoint first
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                # Load from checkpoint
                model_type = checkpoint.get('model_type', 'standard')
                num_classes = checkpoint.get('num_classes', len(self.class_names))
                model = create_model(model_type, num_classes)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Load as state dict
                model = create_model('standard', len(self.class_names))
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def evaluate_on_dataset(self, data_loader: DataLoader, dataset_name: str = "Test"):
        """Evaluate model on a specific dataset."""
        print(f"Evaluating on {dataset_name} dataset...")
        
        predictions, targets, probabilities = self.evaluator.evaluate(data_loader)
        
        # Calculate metrics
        accuracy = np.mean(predictions == targets)
        
        from sklearn.metrics import classification_report, confusion_matrix
        classification_rep = classification_report(targets, predictions, 
                                                 target_names=self.class_names, output_dict=True)
        
        # Print results
        print(f"\n{dataset_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(targets, predictions, target_names=self.class_names))
        
        return {
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities,
            'accuracy': accuracy,
            'classification_report': classification_rep
        }
    
    def create_evaluation_visualizations(self, results: dict, output_dir: str):
        """Create comprehensive evaluation visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        predictions = results['predictions']
        targets = results['targets']
        
        # Confusion matrix
        self.evaluator.plot_confusion_matrix(
            targets, predictions,
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
        
        # Activity distribution comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True distribution
        true_counts = np.bincount(targets, minlength=len(self.class_names))
        ax1.bar(self.class_names, true_counts, color='skyblue', alpha=0.7)
        ax1.set_title('True Activity Distribution')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Predicted distribution
        pred_counts = np.bincount(predictions, minlength=len(self.class_names))
        ax2.bar(self.class_names, pred_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted Activity Distribution')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'activity_distribution_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Per-class performance
        classification_rep = results['classification_report']
        metrics = ['precision', 'recall', 'f1-score']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.class_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [classification_rep[class_name][metric] for class_name in self.class_names]
            ax.bar(x + i * width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Activity Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.class_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Interactive visualizations
        self.visualizer.plot_activity_distribution(
            predictions,
            save_path=os.path.join(output_dir, 'activity_distribution.html')
        )
        
        self.visualizer.plot_confusion_matrix(
            targets, predictions,
            save_path=os.path.join(output_dir, 'confusion_matrix.html')
        )
        
        # Create evaluation dashboard
        dashboard_path = self.visualizer.create_dashboard(
            predictions, targets,
            save_path=os.path.join(output_dir, 'evaluation_dashboard.html')
        )
        
        return dashboard_path
    
    def analyze_prediction_confidence(self, results: dict, output_dir: str):
        """Analyze prediction confidence scores."""
        probabilities = results['probabilities']
        predictions = results['predictions']
        targets = results['targets']
        
        # Calculate confidence scores (max probability)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Analyze confidence by correctness
        correct_predictions = (predictions == targets)
        correct_confidences = confidence_scores[correct_predictions]
        incorrect_confidences = confidence_scores[~correct_predictions]
        
        # Create confidence analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall confidence distribution
        ax1.hist(confidence_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Overall Confidence Score Distribution')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Confidence by correctness
        ax2.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green')
        ax2.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red')
        ax2.set_title('Confidence Score by Prediction Correctness')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Confidence vs accuracy
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                accuracy = np.mean(correct_predictions[mask])
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        ax3.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=6)
        ax3.set_title('Confidence Score vs Accuracy')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Per-class confidence
        class_confidences = []
        for class_idx in range(len(self.class_names)):
            class_mask = (predictions == class_idx)
            if np.sum(class_mask) > 0:
                class_conf = np.mean(confidence_scores[class_mask])
                class_confidences.append(class_conf)
            else:
                class_confidences.append(0)
        
        ax4.bar(self.class_names, class_confidences, color='orange', alpha=0.7)
        ax4.set_title('Average Confidence by Activity Class')
        ax4.set_xlabel('Activity Class')
        ax4.set_ylabel('Average Confidence')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save confidence analysis results
        confidence_analysis = {
            'overall_confidence': {
                'mean': float(np.mean(confidence_scores)),
                'std': float(np.std(confidence_scores)),
                'min': float(np.min(confidence_scores)),
                'max': float(np.max(confidence_scores))
            },
            'correct_predictions_confidence': {
                'mean': float(np.mean(correct_confidences)),
                'std': float(np.std(correct_confidences))
            },
            'incorrect_predictions_confidence': {
                'mean': float(np.mean(incorrect_confidences)),
                'std': float(np.std(incorrect_confidences))
            },
            'per_class_confidence': {
                class_name: float(conf) for class_name, conf in zip(self.class_names, class_confidences)
            }
        }
        
        with open(os.path.join(output_dir, 'confidence_analysis.json'), 'w') as f:
            json.dump(confidence_analysis, f, indent=2)
        
        return confidence_analysis
    
    def run_comprehensive_evaluation(self, data_dir: str, output_dir: str = "evaluation_output"):
        """Run comprehensive model evaluation."""
        print("=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        # Create data loaders
        _, _, test_loader = create_data_loaders(data_dir)
        
        # Evaluate on test set
        test_results = self.evaluate_on_dataset(test_loader, "Test")
        
        # Create visualizations
        dashboard_path = self.create_evaluation_visualizations(test_results, output_dir)
        
        # Analyze confidence
        confidence_analysis = self.analyze_prediction_confidence(test_results, output_dir)
        
        # Save comprehensive results
        comprehensive_results = {
            'test_results': test_results,
            'confidence_analysis': confidence_analysis,
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path
        }
        
        with open(os.path.join(output_dir, 'comprehensive_evaluation.json'), 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print("=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {output_dir}")
        print(f"Dashboard available at: {dashboard_path}")
        
        return comprehensive_results

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Worker Activity Monitoring CNN')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='dummy_data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(args.data_dir, args.output_dir)
    
    return results

if __name__ == "__main__":
    results = main()
