import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import torch
from datetime import datetime, timedelta
import json

class ActivityVisualizer:
    """
    Comprehensive visualization class for worker activity monitoring results.
    """
    
    def __init__(self, class_names: List[str] = None, style: str = 'seaborn-v0_8'):
        self.class_names = class_names or [
            'sitting', 'standing', 'walking', 'working', 
            'resting', 'moving_objects', 'using_tools'
        ]
        self.style = style
        plt.style.use(style)
        
        # Color palette for activities
        self.colors = px.colors.qualitative.Set3[:len(self.class_names)]
        self.color_map = {name: color for name, color in zip(self.class_names, self.colors)}
    
    def plot_activity_distribution(self, predictions: List[int], 
                                 save_path: Optional[str] = None,
                                 title: str = "Activity Distribution") -> go.Figure:
        """Create pie chart showing activity distribution."""
        # Count predictions
        unique, counts = np.unique(predictions, return_counts=True)
        activity_counts = {self.class_names[i]: count for i, count in zip(unique, counts)}
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(activity_counts.keys()),
            values=list(activity_counts.values()),
            marker_colors=[self.color_map[label] for label in activity_counts.keys()],
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title=title,
            font_size=12,
            showlegend=True,
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_activity_timeline(self, predictions: List[int], 
                              timestamps: List[datetime] = None,
                              window_size: int = 10,
                              save_path: Optional[str] = None) -> go.Figure:
        """Create timeline showing activity changes over time."""
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(minutes=i) for i in range(len(predictions))]
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'activity': [self.class_names[p] for p in predictions],
            'activity_id': predictions
        })
        
        # Create timeline plot
        fig = go.Figure()
        
        for i, activity in enumerate(self.class_names):
            activity_data = df[df['activity'] == activity]
            if not activity_data.empty:
                fig.add_trace(go.Scatter(
                    x=activity_data['timestamp'],
                    y=[i] * len(activity_data),
                    mode='markers',
                    name=activity,
                    marker=dict(
                        color=self.color_map[activity],
                        size=8,
                        symbol='circle'
                    ),
                    hovertemplate=f'<b>{activity}</b><br>Time: %{{x}}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Activity Timeline",
            xaxis_title="Time",
            yaxis_title="Activity",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(self.class_names))),
                ticktext=self.class_names
            ),
            height=400,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_activity_heatmap(self, predictions: List[int], 
                             time_windows: int = 24,
                             save_path: Optional[str] = None) -> go.Figure:
        """Create heatmap showing activity patterns over time windows."""
        # Reshape predictions into time windows
        window_size = len(predictions) // time_windows
        if window_size == 0:
            window_size = 1
            time_windows = len(predictions)
        
        # Create activity matrix
        activity_matrix = np.zeros((len(self.class_names), time_windows))
        
        for i in range(time_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(predictions))
            window_predictions = predictions[start_idx:end_idx]
            
            for pred in window_predictions:
                activity_matrix[pred, i] += 1
        
        # Normalize by window size
        activity_matrix = activity_matrix / window_size
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=activity_matrix,
            x=[f'Window {i+1}' for i in range(time_windows)],
            y=self.class_names,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Activity: %{y}<br>Time Window: %{x}<br>Frequency: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Activity Heatmap Over Time",
            xaxis_title="Time Windows",
            yaxis_title="Activities",
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int],
                            save_path: Optional[str] = None) -> go.Figure:
        """Create interactive confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=self.class_names,
            y=self.class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Percentage: %{z:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float],
                           train_accs: List[float], val_accs: List[float],
                           save_path: Optional[str] = None) -> go.Figure:
        """Create training curves plot."""
        epochs = list(range(1, len(train_losses) + 1))
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training and Validation Loss', 'Training and Validation Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, name='Training Loss', 
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_losses, name='Validation Loss',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=train_accs, name='Training Accuracy',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_accs, name='Validation Accuracy',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        
        fig.update_layout(
            title="Training Progress",
            height=400,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_activity_statistics(self, predictions: List[int],
                               save_path: Optional[str] = None) -> go.Figure:
        """Create comprehensive activity statistics dashboard."""
        # Calculate statistics
        unique, counts = np.unique(predictions, return_counts=True)
        activity_stats = {
            'activity': [self.class_names[i] for i in unique],
            'count': counts,
            'percentage': (counts / len(predictions)) * 100,
            'color': [self.color_map[self.class_names[i]] for i in unique]
        }
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Activity Count', 'Activity Percentage', 
                          'Activity Duration', 'Activity Frequency'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Bar chart for counts
        fig.add_trace(
            go.Bar(x=activity_stats['activity'], y=activity_stats['count'],
                  marker_color=activity_stats['color'], name='Count'),
            row=1, col=1
        )
        
        # Pie chart for percentages
        fig.add_trace(
            go.Pie(labels=activity_stats['activity'], values=activity_stats['percentage'],
                  marker_colors=activity_stats['color'], name='Percentage'),
            row=1, col=2
        )
        
        # Duration analysis (simplified)
        durations = [count * 0.1 for count in activity_stats['count']]  # Assuming 0.1 min per prediction
        fig.add_trace(
            go.Bar(x=activity_stats['activity'], y=durations,
                  marker_color=activity_stats['color'], name='Duration (min)'),
            row=2, col=1
        )
        
        # Frequency analysis
        frequencies = [count / len(predictions) for count in activity_stats['count']]
        fig.add_trace(
            go.Scatter(x=activity_stats['activity'], y=frequencies,
                      mode='markers+lines', marker_color=activity_stats['color'],
                      name='Frequency'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Activity Statistics Dashboard",
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_real_time_monitoring(self, predictions: List[int],
                                confidence_scores: List[float] = None,
                                save_path: Optional[str] = None) -> go.Figure:
        """Create real-time monitoring dashboard."""
        if confidence_scores is None:
            confidence_scores = [0.8] * len(predictions)  # Default confidence
        
        timestamps = [datetime.now() - timedelta(minutes=len(predictions)-i) 
                     for i in range(len(predictions))]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Activity Predictions Over Time', 'Confidence Scores'),
            vertical_spacing=0.1
        )
        
        # Activity predictions
        for i, activity in enumerate(self.class_names):
            activity_indices = [j for j, pred in enumerate(predictions) if pred == i]
            if activity_indices:
                activity_times = [timestamps[j] for j in activity_indices]
                activity_confidences = [confidence_scores[j] for j in activity_indices]
                
                fig.add_trace(
                    go.Scatter(x=activity_times, y=[i] * len(activity_times),
                              mode='markers', name=activity,
                              marker=dict(
                                  color=self.color_map[activity],
                                  size=8,
                                  opacity=0.7
                              ),
                              hovertemplate=f'<b>{activity}</b><br>Time: %{{x}}<br>Confidence: %{{customdata:.2f}}<extra></extra>',
                              customdata=activity_confidences),
                    row=1, col=1
                )
        
        # Confidence scores
        fig.add_trace(
            go.Scatter(x=timestamps, y=confidence_scores,
                      mode='lines+markers', name='Confidence',
                      line=dict(color='red', width=2),
                      marker=dict(size=4)),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Activity", 
                        tickmode='array',
                        tickvals=list(range(len(self.class_names))),
                        ticktext=self.class_names,
                        row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        
        fig.update_layout(
            title="Real-time Activity Monitoring",
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dashboard(self, predictions: List[int], 
                        y_true: List[int] = None,
                        train_losses: List[float] = None,
                        val_losses: List[float] = None,
                        train_accs: List[float] = None,
                        val_accs: List[float] = None,
                        save_path: str = "activity_dashboard.html") -> str:
        """Create comprehensive dashboard with all visualizations."""
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Worker Activity Monitoring Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin: 20px 0; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <h1>Worker Activity Monitoring Dashboard</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Activity Distribution</h2>
                    <div id="activity-distribution"></div>
                </div>
                
                <div class="section">
                    <h2>Activity Timeline</h2>
                    <div id="activity-timeline"></div>
                </div>
                
                <div class="section">
                    <h2>Activity Statistics</h2>
                    <div id="activity-statistics"></div>
                </div>
        """
        
        if y_true is not None:
            dashboard_html += """
                <div class="section">
                    <h2>Confusion Matrix</h2>
                    <div id="confusion-matrix"></div>
                </div>
            """
        
        if train_losses is not None:
            dashboard_html += """
                <div class="section">
                    <h2>Training Progress</h2>
                    <div id="training-curves"></div>
                </div>
            """
        
        dashboard_html += """
            </div>
            
            <script>
        """
        
        # Add JavaScript for plots
        dashboard_html += f"""
                // Activity Distribution
                var activityDist = {self.plot_activity_distribution(predictions).to_json()};
                Plotly.newPlot('activity-distribution', activityDist.data, activityDist.layout);
                
                // Activity Timeline
                var activityTimeline = {self.plot_activity_timeline(predictions).to_json()};
                Plotly.newPlot('activity-timeline', activityTimeline.data, activityTimeline.layout);
                
                // Activity Statistics
                var activityStats = {self.plot_activity_statistics(predictions).to_json()};
                Plotly.newPlot('activity-statistics', activityStats.data, activityStats.layout);
        """
        
        if y_true is not None:
            dashboard_html += f"""
                // Confusion Matrix
                var confusionMatrix = {self.plot_confusion_matrix(y_true, predictions).to_json()};
                Plotly.newPlot('confusion-matrix', confusionMatrix.data, confusionMatrix.layout);
            """
        
        if train_losses is not None:
            dashboard_html += f"""
                // Training Curves
                var trainingCurves = {self.plot_training_curves(train_losses, val_losses, train_accs, val_accs).to_json()};
                Plotly.newPlot('training-curves', trainingCurves.data, trainingCurves.layout);
            """
        
        dashboard_html += """
            </script>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(dashboard_html)
        
        print(f"Dashboard saved to {save_path}")
        return save_path

def create_sample_visualizations():
    """Create sample visualizations for demonstration."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Sample predictions (biased towards some activities)
    predictions = np.random.choice(7, n_samples, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05])
    
    # Sample true labels
    y_true = predictions.copy()
    # Add some noise to simulate prediction errors
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y_true[noise_indices] = np.random.choice(7, len(noise_indices))
    
    # Sample training data
    train_losses = [2.0 * np.exp(-i/20) + 0.1 + 0.1*np.random.random() for i in range(50)]
    val_losses = [1.8 * np.exp(-i/25) + 0.15 + 0.1*np.random.random() for i in range(50)]
    train_accs = [100 * (1 - loss/2) for loss in train_losses]
    val_accs = [100 * (1 - loss/2) for loss in val_losses]
    
    # Create visualizer
    visualizer = ActivityVisualizer()
    
    # Generate all visualizations
    print("Creating sample visualizations...")
    
    # Individual plots
    visualizer.plot_activity_distribution(predictions, "activity_distribution.html")
    visualizer.plot_activity_timeline(predictions, "activity_timeline.html")
    visualizer.plot_activity_heatmap(predictions, "activity_heatmap.html")
    visualizer.plot_confusion_matrix(y_true, predictions, "confusion_matrix.html")
    visualizer.plot_training_curves(train_losses, val_losses, train_accs, val_accs, "training_curves.html")
    visualizer.plot_activity_statistics(predictions, "activity_statistics.html")
    visualizer.plot_real_time_monitoring(predictions, "real_time_monitoring.html")
    
    # Comprehensive dashboard
    dashboard_path = visualizer.create_dashboard(
        predictions, y_true, train_losses, val_losses, train_accs, val_accs
    )
    
    print(f"All visualizations created successfully!")
    print(f"Open {dashboard_path} in your browser to view the complete dashboard.")

if __name__ == "__main__":
    create_sample_visualizations()
