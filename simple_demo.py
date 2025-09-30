#!/usr/bin/env python3
"""
Simple demo for Worker Activity Monitoring with CNN - Static Output Only.
This version creates only static images and console output, no HTML files.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample activity data...")
    
    # Generate sample predictions (biased towards some activities)
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic activity distribution
    activity_probs = [0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05]  # sitting, standing, walking, working, resting, moving_objects, using_tools
    predictions = np.random.choice(7, n_samples, p=activity_probs)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=8)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # Generate confidence scores
    confidence_scores = np.random.beta(2, 2, n_samples)  # Beta distribution for realistic confidence scores
    
    return predictions, timestamps, confidence_scores

def print_activity_summary(predictions, class_names):
    """Print activity summary to console."""
    print("\n" + "="*60)
    print("WORKER ACTIVITY SUMMARY")
    print("="*60)
    
    unique, counts = np.unique(predictions, return_counts=True)
    total = len(predictions)
    
    print(f"Total observations: {total}")
    print(f"Time period: 8 hours")
    print(f"Average observation interval: {8*60/total:.1f} minutes")
    print("\nActivity Distribution:")
    print("-" * 40)
    
    for i, count in zip(unique, counts):
        percentage = (count / total) * 100
        bar = "*" * int(percentage / 2)  # Simple text bar
        print(f"{class_names[i]:<15}: {count:>4} ({percentage:>5.1f}%) {bar}")
    
    # Find most/least common activities
    most_common_idx = np.argmax(counts)
    least_common_idx = np.argmin(counts)
    
    print(f"\nMost common activity: {class_names[unique[most_common_idx]]} ({counts[most_common_idx]} occurrences)")
    print(f"Least common activity: {class_names[unique[least_common_idx]]} ({counts[least_common_idx]} occurrences)")

def create_static_visualizations(predictions, timestamps, confidence_scores):
    """Create static matplotlib visualizations only."""
    class_names = ['sitting', 'standing', 'walking', 'working', 'resting', 'moving_objects', 'using_tools']
    
    # Create output directory
    output_dir = "static_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating static visualizations in: {output_dir}")
    
    # 1. Activity Distribution Bar Chart
    plt.figure(figsize=(12, 6))
    unique, counts = np.unique(predictions, return_counts=True)
    activity_counts = [counts[i] if i in unique else 0 for i in range(len(class_names))]
    
    colors = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462', '#B3DE69']
    bars = plt.bar(class_names, activity_counts, color=colors)
    
    plt.title('Worker Activity Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Activity Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, activity_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "activity_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Confidence Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Confidence Score Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
               label=f'Mean: {np.mean(confidence_scores):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Activity Timeline
    plt.figure(figsize=(15, 8))
    color_map = {
        'sitting': '#8DD3C7',
        'standing': '#FFFFB3', 
        'walking': '#BEBADA',
        'working': '#FB8072',
        'resting': '#80B1D3',
        'moving_objects': '#FDB462',
        'using_tools': '#B3DE69'
    }
    
    for i, activity in enumerate(class_names):
        activity_indices = [j for j, pred in enumerate(predictions) if pred == i]
        if activity_indices:
            activity_times = [timestamps[j] for j in activity_indices]
            plt.scatter(activity_times, [i] * len(activity_times), 
                       label=activity, alpha=0.6, s=20, color=color_map[activity])
    
    plt.title('Worker Activity Timeline', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Activity', fontsize=12)
    plt.yticks(range(len(class_names)), class_names)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "activity_timeline.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Activity Heatmap (Hourly)
    plt.figure(figsize=(12, 6))
    
    # Create hourly activity matrix
    hours = 8
    activity_matrix = np.zeros((len(class_names), hours))
    
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour - timestamps[0].hour
        if 0 <= hour < hours:
            activity_matrix[predictions[i], hour] += 1
    
    # Normalize by hour
    for hour in range(hours):
        total = np.sum(activity_matrix[:, hour])
        if total > 0:
            activity_matrix[:, hour] /= total
    
    plt.imshow(activity_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Activity Frequency')
    plt.title('Activity Heatmap by Hour', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Activity Type', fontsize=12)
    plt.yticks(range(len(class_names)), class_names)
    plt.xticks(range(hours), [f'Hour {i+1}' for i in range(hours)])
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(hours):
            text = plt.text(j, i, f'{activity_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "activity_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_dir

def run_simple_demo():
    """Run the simple demo with static output only."""
    print("=" * 80)
    print("WORKER ACTIVITY MONITORING - SIMPLE DEMO (STATIC OUTPUT)")
    print("=" * 80)
    
    # Create sample data
    predictions, timestamps, confidence_scores = create_sample_data()
    class_names = ['sitting', 'standing', 'walking', 'working', 'resting', 'moving_objects', 'using_tools']
    
    # Print summary to console
    print_activity_summary(predictions, class_names)
    
    # Create static visualizations
    output_dir = create_static_visualizations(predictions, timestamps, confidence_scores)
    
    print("\n" + "=" * 80)
    print("SIMPLE DEMO COMPLETED!")
    print("=" * 80)
    print(f"Static visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("- activity_distribution.png (Bar chart)")
    print("- confidence_distribution.png (Histogram)")
    print("- activity_timeline.png (Timeline plot)")
    print("- activity_heatmap.png (Heatmap)")
    print("\nNo HTML files created - only static images and console output!")
    print("=" * 80)

def main():
    """Main function."""
    try:
        run_simple_demo()
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
