import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import List, Dict, Optional
from graphs import ActivityVisualizer

class ActivityDashboard:
    """
    Interactive Dash dashboard for worker activity monitoring.
    """
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or [
            'sitting', 'standing', 'walking', 'working', 
            'resting', 'moving_objects', 'using_tools'
        ]
        self.visualizer = ActivityVisualizer(self.class_names)
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Worker Activity Monitoring Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Control panel
            html.Div([
                html.Div([
                    html.Label("Select Time Range:"),
                    dcc.Dropdown(
                        id='time-range-dropdown',
                        options=[
                            {'label': 'Last Hour', 'value': '1h'},
                            {'label': 'Last 4 Hours', 'value': '4h'},
                            {'label': 'Last 8 Hours', 'value': '8h'},
                            {'label': 'Last 24 Hours', 'value': '24h'},
                            {'label': 'All Time', 'value': 'all'}
                        ],
                        value='4h',
                        style={'width': '200px'}
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Update Interval (seconds):"),
                    dcc.Slider(
                        id='update-interval-slider',
                        min=1,
                        max=60,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(0, 61, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '40%', 'display': 'inline-block', 'marginLeft': '5%'}),
                
                html.Div([
                    html.Button('Refresh Data', id='refresh-button', n_clicks=0,
                              style={'backgroundColor': '#007bff', 'color': 'white', 
                                   'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px'})
                ], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '5%'})
            ], style={'marginBottom': 30, 'padding': '20px', 'backgroundColor': '#f8f9fa', 
                     'borderRadius': '10px'}),
            
            # Real-time monitoring section
            html.Div([
                html.H2("Real-time Activity Monitoring"),
                dcc.Graph(id='real-time-activity-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=5000,  # Update every 5 seconds
                    n_intervals=0
                )
            ], style={'marginBottom': 30}),
            
            # Activity distribution section
            html.Div([
                html.H2("Activity Distribution"),
                html.Div([
                    dcc.Graph(id='activity-distribution-pie', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='activity-distribution-bar', style={'width': '50%', 'display': 'inline-block'})
                ])
            ], style={'marginBottom': 30}),
            
            # Activity timeline section
            html.Div([
                html.H2("Activity Timeline"),
                dcc.Graph(id='activity-timeline-graph')
            ], style={'marginBottom': 30}),
            
            # Activity statistics section
            html.Div([
                html.H2("Activity Statistics"),
                html.Div([
                    dcc.Graph(id='activity-heatmap', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='activity-frequency', style={'width': '50%', 'display': 'inline-block'})
                ])
            ], style={'marginBottom': 30}),
            
            # Performance metrics section
            html.Div([
                html.H2("Model Performance"),
                html.Div([
                    dcc.Graph(id='confusion-matrix', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='performance-metrics', style={'width': '50%', 'display': 'inline-block'})
                ])
            ], style={'marginBottom': 30}),
            
            # Data table section
            html.Div([
                html.H2("Recent Activity Log"),
                html.Div(id='activity-table')
            ])
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('real-time-activity-graph', 'figure'),
             Output('activity-distribution-pie', 'figure'),
             Output('activity-distribution-bar', 'figure'),
             Output('activity-timeline-graph', 'figure'),
             Output('activity-heatmap', 'figure'),
             Output('activity-frequency', 'figure'),
             Output('confusion-matrix', 'figure'),
             Output('performance-metrics', 'figure'),
             Output('activity-table', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks'),
             Input('time-range-dropdown', 'value')]
        )
        def update_dashboard(n_intervals, n_clicks, time_range):
            """Update all dashboard components."""
            # Generate sample data (in real implementation, this would come from your model)
            predictions, y_true, timestamps = self.generate_sample_data(time_range)
            
            # Update all graphs
            real_time_fig = self.create_real_time_graph(predictions, timestamps)
            pie_fig = self.create_activity_pie_chart(predictions)
            bar_fig = self.create_activity_bar_chart(predictions)
            timeline_fig = self.create_timeline_graph(predictions, timestamps)
            heatmap_fig = self.create_heatmap(predictions)
            frequency_fig = self.create_frequency_graph(predictions)
            confusion_fig = self.create_confusion_matrix(y_true, predictions)
            performance_fig = self.create_performance_metrics(y_true, predictions)
            table = self.create_activity_table(predictions, timestamps)
            
            return (real_time_fig, pie_fig, bar_fig, timeline_fig, 
                   heatmap_fig, frequency_fig, confusion_fig, performance_fig, table)
    
    def generate_sample_data(self, time_range: str) -> tuple:
        """Generate sample data for demonstration."""
        # Determine number of samples based on time range
        if time_range == '1h':
            n_samples = 60
        elif time_range == '4h':
            n_samples = 240
        elif time_range == '8h':
            n_samples = 480
        elif time_range == '24h':
            n_samples = 1440
        else:  # all
            n_samples = 2000
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=n_samples)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
        
        # Generate predictions (biased towards some activities)
        np.random.seed(42)
        predictions = np.random.choice(7, n_samples, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05])
        
        # Generate true labels with some noise
        y_true = predictions.copy()
        noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        y_true[noise_indices] = np.random.choice(7, len(noise_indices))
        
        return predictions, y_true, timestamps
    
    def create_real_time_graph(self, predictions: List[int], timestamps: List[datetime]) -> go.Figure:
        """Create real-time activity monitoring graph."""
        fig = go.Figure()
        
        # Add traces for each activity
        for i, activity in enumerate(self.class_names):
            activity_indices = [j for j, pred in enumerate(predictions) if pred == i]
            if activity_indices:
                activity_times = [timestamps[j] for j in activity_indices]
                
                fig.add_trace(go.Scatter(
                    x=activity_times,
                    y=[i] * len(activity_times),
                    mode='markers',
                    name=activity,
                    marker=dict(
                        color=self.visualizer.color_map[activity],
                        size=8,
                        opacity=0.7
                    ),
                    hovertemplate=f'<b>{activity}</b><br>Time: %{{x}}<extra></extra>'
                ))
        
        fig.update_layout(
            title="Real-time Activity Monitoring",
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
        
        return fig
    
    def create_activity_pie_chart(self, predictions: List[int]) -> go.Figure:
        """Create activity distribution pie chart."""
        unique, counts = np.unique(predictions, return_counts=True)
        activity_counts = {self.class_names[i]: count for i, count in zip(unique, counts)}
        
        fig = go.Figure(data=[go.Pie(
            labels=list(activity_counts.keys()),
            values=list(activity_counts.values()),
            marker_colors=[self.visualizer.color_map[label] for label in activity_counts.keys()],
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Activity Distribution",
            height=400
        )
        
        return fig
    
    def create_activity_bar_chart(self, predictions: List[int]) -> go.Figure:
        """Create activity distribution bar chart."""
        unique, counts = np.unique(predictions, return_counts=True)
        activity_counts = {self.class_names[i]: count for i, count in zip(unique, counts)}
        
        fig = go.Figure(data=[go.Bar(
            x=list(activity_counts.keys()),
            y=list(activity_counts.values()),
            marker_color=[self.visualizer.color_map[label] for label in activity_counts.keys()]
        )])
        
        fig.update_layout(
            title="Activity Count",
            xaxis_title="Activity",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    def create_timeline_graph(self, predictions: List[int], timestamps: List[datetime]) -> go.Figure:
        """Create activity timeline graph."""
        return self.visualizer.plot_activity_timeline(predictions, timestamps)
    
    def create_heatmap(self, predictions: List[int]) -> go.Figure:
        """Create activity heatmap."""
        return self.visualizer.plot_activity_heatmap(predictions)
    
    def create_frequency_graph(self, predictions: List[int]) -> go.Figure:
        """Create activity frequency graph."""
        unique, counts = np.unique(predictions, return_counts=True)
        frequencies = counts / len(predictions)
        
        fig = go.Figure(data=[go.Bar(
            x=[self.class_names[i] for i in unique],
            y=frequencies,
            marker_color=[self.visualizer.color_map[self.class_names[i]] for i in unique]
        )])
        
        fig.update_layout(
            title="Activity Frequency",
            xaxis_title="Activity",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig
    
    def create_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> go.Figure:
        """Create confusion matrix."""
        return self.visualizer.plot_confusion_matrix(y_true, y_pred)
    
    def create_performance_metrics(self, y_true: List[int], y_pred: List[int]) -> go.Figure:
        """Create performance metrics graph."""
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics_df = pd.DataFrame({
            'Activity': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='Precision', x=metrics_df['Activity'], y=metrics_df['Precision']))
        fig.add_trace(go.Bar(name='Recall', x=metrics_df['Activity'], y=metrics_df['Recall']))
        fig.add_trace(go.Bar(name='F1-Score', x=metrics_df['Activity'], y=metrics_df['F1-Score']))
        
        fig.update_layout(
            title=f"Performance Metrics (Overall Accuracy: {accuracy:.3f})",
            xaxis_title="Activity",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_activity_table(self, predictions: List[int], timestamps: List[datetime]) -> html.Div:
        """Create activity log table."""
        # Get last 20 predictions
        recent_data = list(zip(timestamps[-20:], predictions[-20:]))
        
        table_rows = []
        for timestamp, pred in recent_data:
            activity = self.class_names[pred]
            table_rows.append(
                html.Tr([
                    html.Td(timestamp.strftime('%H:%M:%S')),
                    html.Td(activity),
                    html.Td(f"{np.random.random():.3f}")  # Confidence score
                ])
            )
        
        table = html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Time"),
                    html.Th("Activity"),
                    html.Th("Confidence")
                ])
            ]),
            html.Tbody(table_rows)
        ], style={'width': '100%', 'border': '1px solid #ddd'})
        
        return table
    
    def run(self, debug=True, port=8050):
        """Run the dashboard."""
        print(f"Starting dashboard on http://localhost:{port}")
        self.app.run_server(debug=debug, port=port)

def create_sample_dashboard():
    """Create and run a sample dashboard."""
    dashboard = ActivityDashboard()
    dashboard.run(debug=True, port=8050)

if __name__ == "__main__":
    create_sample_dashboard()
