import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import time
import argparse
import os
import json
from datetime import datetime, timedelta
from collections import deque
import threading
import queue
import sys
sys.path.append('..')

from models.cnn_model import create_model
from models.model_utils import load_model
from visualization.graphs import ActivityVisualizer

class RealTimeActivityMonitor:
    """
    Real-time worker activity monitoring system using CNN.
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
        
        # Activity tracking
        self.activity_history = deque(maxlen=1000)  # Keep last 1000 predictions
        self.confidence_history = deque(maxlen=1000)
        self.timestamp_history = deque(maxlen=1000)
        
        # Real-time monitoring settings
        self.update_interval = self.config.get('update_interval', 1.0)  # seconds
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.smoothing_window = self.config.get('smoothing_window', 5)
        
        print(f"Real-time monitor initialized on {self.device}")
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
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_resized)
        
        # Apply transforms (same as training)
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to tensor and add batch dimension
        tensor = transform(pil_image).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict_activity(self, frame: np.ndarray) -> tuple:
        """Predict activity from frame."""
        with torch.no_grad():
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            
            # Get prediction
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score, probabilities.cpu().numpy()[0]
    
    def smooth_predictions(self, new_prediction: int, new_confidence: float) -> tuple:
        """Apply temporal smoothing to predictions."""
        # Add new prediction to history
        self.activity_history.append(new_prediction)
        self.confidence_history.append(new_confidence)
        self.timestamp_history.append(datetime.now())
        
        # Apply smoothing if we have enough history
        if len(self.activity_history) >= self.smoothing_window:
            # Get recent predictions
            recent_predictions = list(self.activity_history)[-self.smoothing_window:]
            recent_confidences = list(self.confidence_history)[-self.smoothing_window:]
            
            # Weighted voting based on confidence
            weighted_votes = np.zeros(len(self.class_names))
            for pred, conf in zip(recent_predictions, recent_confidences):
                weighted_votes[pred] += conf
            
            # Get smoothed prediction
            smoothed_prediction = np.argmax(weighted_votes)
            smoothed_confidence = np.max(weighted_votes) / np.sum(weighted_votes)
            
            return smoothed_prediction, smoothed_confidence
        
        return new_prediction, new_confidence
    
    def draw_activity_info(self, frame: np.ndarray, prediction: int, confidence: float) -> np.ndarray:
        """Draw activity information on frame."""
        # Get activity name
        activity_name = self.class_names[prediction]
        
        # Create info text
        info_text = f"Activity: {activity_name}"
        confidence_text = f"Confidence: {confidence:.3f}"
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 80), (255, 255, 255), 2)
        
        # Draw text
        cv2.putText(frame, info_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 10
        bar_x = 20
        bar_y = 70
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Confidence bar
        confidence_width = int(bar_width * confidence)
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
        
        return frame
    
    def monitor_webcam(self, camera_id: int = 0, save_video: bool = False, 
                      output_path: str = "monitoring_output.mp4"):
        """Monitor activity from webcam feed."""
        print(f"Starting webcam monitoring (Camera ID: {camera_id})")
        
        # Initialize video capture
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Video writer for saving
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Press 'q' to quit, 's' to save current frame, 'r' to reset history")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Predict activity
                prediction, confidence, probabilities = self.predict_activity(frame)
                
                # Apply smoothing
                smoothed_pred, smoothed_conf = self.smooth_predictions(prediction, confidence)
                
                # Draw information on frame
                frame = self.draw_activity_info(frame, smoothed_pred, smoothed_conf)
                
                # Display frame
                cv2.imshow('Worker Activity Monitoring', frame)
                
                # Save video if enabled
                if save_video:
                    out.write(frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"frame_{timestamp}_{self.class_names[smoothed_pred]}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    # Reset history
                    self.activity_history.clear()
                    self.confidence_history.clear()
                    self.timestamp_history.clear()
                    print("History reset")
        
        except KeyboardInterrupt:
            print("Monitoring interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            print("Monitoring stopped")
    
    def monitor_video_file(self, video_path: str, save_output: bool = False,
                          output_path: str = "video_analysis_output.mp4"):
        """Monitor activity from video file."""
        print(f"Analyzing video file: {video_path}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Video writer for output
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict activity
                prediction, confidence, probabilities = self.predict_activity(frame)
                
                # Apply smoothing
                smoothed_pred, smoothed_conf = self.smooth_predictions(prediction, confidence)
                
                # Draw information on frame
                frame = self.draw_activity_info(frame, smoothed_pred, smoothed_conf)
                
                # Display frame
                cv2.imshow('Video Activity Analysis', frame)
                
                # Save output if enabled
                if save_output:
                    out.write(frame)
                
                # Progress update
                frame_count += 1
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed_time = time.time() - start_time
                    fps_current = frame_count / elapsed_time
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_current:.1f} FPS")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("Analysis interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if save_output:
                out.release()
            cv2.destroyAllWindows()
            
            # Print summary
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            print(f"Analysis completed: {frame_count} frames processed in {total_time:.1f}s ({avg_fps:.1f} FPS)")
    
    def generate_activity_report(self, output_path: str = "activity_report.html"):
        """Generate comprehensive activity report."""
        if not self.activity_history:
            print("No activity data available for report generation")
            return
        
        # Convert history to lists
        activities = list(self.activity_history)
        confidences = list(self.confidence_history)
        timestamps = list(self.timestamp_history)
        
        # Create visualizations
        dashboard_path = self.visualizer.create_dashboard(
            activities, save_path=output_path
        )
        
        # Generate summary statistics
        activity_counts = np.bincount(activities, minlength=len(self.class_names))
        total_time = (timestamps[-1] - timestamps[0]).total_seconds() / 60  # minutes
        
        summary = {
            'total_predictions': len(activities),
            'monitoring_duration_minutes': total_time,
            'activity_distribution': {
                self.class_names[i]: int(count) for i, count in enumerate(activity_counts)
            },
            'average_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences)),
            'most_common_activity': self.class_names[np.argmax(activity_counts)],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = output_path.replace('.html', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Activity report generated: {output_path}")
        print(f"Summary saved: {summary_path}")
        
        return dashboard_path, summary

def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description='Real-time Worker Activity Monitoring')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--source', type=str, default='0', help='Video source (camera ID or video file path)')
    parser.add_argument('--save_video', action='store_true', help='Save output video')
    parser.add_argument('--output_path', type=str, default='monitoring_output.mp4', help='Output video path')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--smoothing_window', type=int, default=5, help='Smoothing window size')
    parser.add_argument('--update_interval', type=float, default=1.0, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'confidence_threshold': args.confidence_threshold,
        'smoothing_window': args.smoothing_window,
        'update_interval': args.update_interval
    }
    
    # Create monitor
    monitor = RealTimeActivityMonitor(args.model_path, config)
    
    # Determine source type
    if args.source.isdigit():
        # Camera
        camera_id = int(args.source)
        monitor.monitor_webcam(camera_id, args.save_video, args.output_path)
    else:
        # Video file
        monitor.monitor_video_file(args.source, args.save_video, args.output_path)
    
    # Generate report
    monitor.generate_activity_report()

if __name__ == "__main__":
    main()
