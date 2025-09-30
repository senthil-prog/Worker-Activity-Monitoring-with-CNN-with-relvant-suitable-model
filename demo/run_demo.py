import os
import sys
import json
import time
import argparse
from datetime import datetime
import subprocess

# Add project root to path
sys.path.append('..')

from demo_data import DemoDataGenerator
from models.cnn_model import create_model, print_model_summary
from training.train import TrainingManager
from inference.predict import BatchActivityPredictor
from visualization.dashboard import ActivityDashboard
from visualization.graphs import ActivityVisualizer

class WorkerActivityDemo:
    """
    Complete demonstration of the Worker Activity Monitoring system.
    """
    
    def __init__(self, output_dir: str = "demo_output"):
        self.output_dir = output_dir
        self.class_names = [
            'sitting', 'standing', 'walking', 'working', 
            'resting', 'moving_objects', 'using_tools'
        ]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 80)
        print("WORKER ACTIVITY MONITORING - CNN DEMO")
        print("=" * 80)
        print(f"Output directory: {output_dir}")
        print(f"Activity classes: {', '.join(self.class_names)}")
        print("=" * 80)
    
    def step1_generate_demo_data(self, samples_per_class: int = 50):
        """Step 1: Generate synthetic demo data."""
        print("\n" + "=" * 60)
        print("STEP 1: GENERATING DEMO DATA")
        print("=" * 60)
        
        data_dir = os.path.join(self.output_dir, "demo_data")
        generator = DemoDataGenerator(data_dir)
        
        # Generate dataset
        dataset_info = generator.generate_dataset(
            samples_per_class=samples_per_class,
            splits={'train': 0.7, 'val': 0.2, 'test': 0.1}
        )
        
        # Create sample images
        generator.create_sample_images(num_samples=3)
        
        print(f"Demo data generated in: {data_dir}")
        return data_dir, dataset_info
    
    def step2_explore_model(self):
        """Step 2: Explore and demonstrate the CNN model."""
        print("\n" + "=" * 60)
        print("STEP 2: CNN MODEL EXPLORATION")
        print("=" * 60)
        
        # Create and display model
        print("Creating CNN model...")
        model = create_model('standard', num_classes=len(self.class_names))
        
        print("\nModel Architecture:")
        print_model_summary(model)
        
        # Save model info
        model_info = {
            'model_type': 'WorkerActivityCNN',
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        model_info_path = os.path.join(self.output_dir, "model_info.json")
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model info saved to: {model_info_path}")
        return model_info
    
    def step3_train_model(self, data_dir: str, epochs: int = 20):
        """Step 3: Train the CNN model."""
        print("\n" + "=" * 60)
        print("STEP 3: TRAINING CNN MODEL")
        print("=" * 60)
        
        # Training configuration
        config = {
            'data_dir': data_dir,
            'output_dir': os.path.join(self.output_dir, "training_output"),
            'batch_size': 16,  # Smaller batch size for demo
            'epochs': epochs,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 5,
            'model_type': 'standard',
            'pretrained': False,
            'image_size': (224, 224),
            'num_workers': 2
        }
        
        print("Training configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Create and run training manager
        trainer = TrainingManager(config)
        results = trainer.run_training_pipeline()
        
        print(f"Training completed! Results saved to: {results['output_dir']}")
        return results
    
    def step4_evaluate_model(self, model_path: str, data_dir: str):
        """Step 4: Evaluate the trained model."""
        print("\n" + "=" * 60)
        print("STEP 4: MODEL EVALUATION")
        print("=" * 60)
        
        # Create predictor for evaluation
        predictor = BatchActivityPredictor(model_path)
        
        # Test on sample images
        test_dir = os.path.join(data_dir, "test")
        if os.path.exists(test_dir):
            print("Evaluating on test set...")
            results = predictor.predict_directory(test_dir)
            
            # Analyze results
            analysis = predictor.analyze_predictions(results)
            
            # Save evaluation results
            eval_output_dir = os.path.join(self.output_dir, "evaluation_output")
            os.makedirs(eval_output_dir, exist_ok=True)
            
            # Save predictions
            predictions_path = os.path.join(eval_output_dir, "test_predictions.json")
            predictor.save_predictions(results, predictions_path)
            
            # Save analysis
            analysis_path = os.path.join(eval_output_dir, "evaluation_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Create visualizations
            dashboard_path = predictor.create_prediction_visualizations(results, eval_output_dir)
            
            print(f"Evaluation completed! Results saved to: {eval_output_dir}")
            print(f"Dashboard available at: {dashboard_path}")
            
            return analysis
        else:
            print("Test directory not found, skipping evaluation")
            return None
    
    def step5_demonstrate_inference(self, model_path: str, data_dir: str):
        """Step 5: Demonstrate inference capabilities."""
        print("\n" + "=" * 60)
        print("STEP 5: INFERENCE DEMONSTRATION")
        print("=" * 60)
        
        # Create predictor
        predictor = BatchActivityPredictor(model_path)
        
        # Test on sample images
        samples_dir = os.path.join(data_dir, "samples")
        if os.path.exists(samples_dir):
            print("Running inference on sample images...")
            results = predictor.predict_directory(samples_dir)
            
            # Display results
            print("\nInference Results:")
            print("-" * 50)
            for result in results[:10]:  # Show first 10 results
                print(f"Image: {os.path.basename(result['image_path'])}")
                print(f"  Predicted Activity: {result['predicted_activity']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Top 3 Probabilities:")
                sorted_probs = sorted(result['class_probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                for i, (activity, prob) in enumerate(sorted_probs[:3]):
                    print(f"    {i+1}. {activity}: {prob:.3f}")
                print()
            
            # Save inference results
            inference_output_dir = os.path.join(self.output_dir, "inference_output")
            os.makedirs(inference_output_dir, exist_ok=True)
            
            predictions_path = os.path.join(inference_output_dir, "sample_predictions.json")
            predictor.save_predictions(results, predictions_path)
            
            # Create visualizations
            dashboard_path = predictor.create_prediction_visualizations(results, inference_output_dir)
            
            print(f"Inference results saved to: {inference_output_dir}")
            print(f"Dashboard available at: {dashboard_path}")
            
            return results
        else:
            print("Samples directory not found, skipping inference demo")
            return None
    
    def step6_create_visualizations(self, training_results: dict = None, 
                                  evaluation_results: dict = None,
                                  inference_results: list = None):
        """Step 6: Create comprehensive visualizations."""
        print("\n" + "=" * 60)
        print("STEP 6: CREATING VISUALIZATIONS")
        print("=" * 60)
        
        viz_output_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_output_dir, exist_ok=True)
        
        visualizer = ActivityVisualizer(self.class_names)
        
        # Create sample data for visualization
        import numpy as np
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample predictions
        sample_predictions = np.random.choice(len(self.class_names), n_samples, 
                                            p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05])
        
        # Create various visualizations
        print("Creating activity distribution chart...")
        visualizer.plot_activity_distribution(
            sample_predictions,
            save_path=os.path.join(viz_output_dir, "activity_distribution.html")
        )
        
        print("Creating activity timeline...")
        visualizer.plot_activity_timeline(
            sample_predictions,
            save_path=os.path.join(viz_output_dir, "activity_timeline.html")
        )
        
        print("Creating activity heatmap...")
        visualizer.plot_activity_heatmap(
            sample_predictions,
            save_path=os.path.join(viz_output_dir, "activity_heatmap.html")
        )
        
        print("Creating activity statistics dashboard...")
        visualizer.plot_activity_statistics(
            sample_predictions,
            save_path=os.path.join(viz_output_dir, "activity_statistics.html")
        )
        
        # Create comprehensive dashboard
        print("Creating comprehensive dashboard...")
        dashboard_path = visualizer.create_dashboard(
            sample_predictions,
            save_path=os.path.join(viz_output_dir, "comprehensive_dashboard.html")
        )
        
        print(f"All visualizations saved to: {viz_output_dir}")
        print(f"Main dashboard available at: {dashboard_path}")
        
        return dashboard_path
    
    def step7_launch_interactive_dashboard(self):
        """Step 7: Launch interactive dashboard."""
        print("\n" + "=" * 60)
        print("STEP 7: INTERACTIVE DASHBOARD")
        print("=" * 60)
        
        print("Creating interactive dashboard...")
        dashboard = ActivityDashboard(self.class_names)
        
        print("Interactive dashboard created!")
        print("To launch the dashboard, run:")
        print("  python visualization/dashboard.py")
        print("Then open http://localhost:8050 in your browser")
        
        return dashboard
    
    def run_complete_demo(self, samples_per_class: int = 50, epochs: int = 20):
        """Run the complete demonstration pipeline."""
        start_time = time.time()
        
        try:
            # Step 1: Generate demo data
            data_dir, dataset_info = self.step1_generate_demo_data(samples_per_class)
            
            # Step 2: Explore model
            model_info = self.step2_explore_model()
            
            # Step 3: Train model
            training_results = self.step3_train_model(data_dir, epochs)
            model_path = os.path.join(training_results['output_dir'], 'model_checkpoint.pth')
            
            # Step 4: Evaluate model
            evaluation_results = self.step4_evaluate_model(model_path, data_dir)
            
            # Step 5: Demonstrate inference
            inference_results = self.step5_demonstrate_inference(model_path, data_dir)
            
            # Step 6: Create visualizations
            dashboard_path = self.step6_create_visualizations(
                training_results, evaluation_results, inference_results
            )
            
            # Step 7: Interactive dashboard info
            self.step7_launch_interactive_dashboard()
            
            # Create demo summary
            demo_summary = {
                'demo_timestamp': datetime.now().isoformat(),
                'total_duration_minutes': (time.time() - start_time) / 60,
                'samples_per_class': samples_per_class,
                'epochs': epochs,
                'class_names': self.class_names,
                'output_directory': self.output_dir,
                'dataset_info': dataset_info,
                'model_info': model_info,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'dashboard_path': dashboard_path
            }
            
            # Save demo summary
            summary_path = os.path.join(self.output_dir, "demo_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(demo_summary, f, indent=2)
            
            print("\n" + "=" * 80)
            print("DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"Total duration: {(time.time() - start_time) / 60:.1f} minutes")
            print(f"All results saved to: {self.output_dir}")
            print(f"Main dashboard: {dashboard_path}")
            print(f"Demo summary: {summary_path}")
            print("\nNext steps:")
            print("1. Open the dashboard HTML files in your browser")
            print("2. Run 'python visualization/dashboard.py' for interactive dashboard")
            print("3. Use 'python inference/monitor.py' for real-time monitoring")
            print("4. Use 'python inference/predict.py' for batch prediction")
            print("=" * 80)
            
            return demo_summary
            
        except Exception as e:
            print(f"\nDemo failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Worker Activity Monitoring CNN Demo')
    parser.add_argument('--output_dir', type=str, default='demo_output', help='Output directory')
    parser.add_argument('--samples_per_class', type=int, default=50, help='Samples per class')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--quick', action='store_true', help='Quick demo with fewer samples and epochs')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick demo
    if args.quick:
        args.samples_per_class = 20
        args.epochs = 5
        print("Running quick demo with reduced parameters...")
    
    # Create and run demo
    demo = WorkerActivityDemo(args.output_dir)
    results = demo.run_complete_demo(args.samples_per_class, args.epochs)
    
    if results:
        print("Demo completed successfully!")
    else:
        print("Demo failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
