#!/usr/bin/env python3
"""
Show All Graphs - Worker Activity Monitoring
This script displays all visualization graphs exactly once in sequence.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample activity data...")
    
    # Generate sample predictions (biased towards some activities)
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic activity distribution
    activity_probs = [0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05]
    predictions = np.random.choice(7, n_samples, p=activity_probs)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=8)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # Generate confidence scores
    confidence_scores = np.random.beta(2, 2, n_samples)
    
    return predictions, timestamps, confidence_scores

def show_graph_1_activity_distribution(predictions, class_names):
    """Show Graph 1: Activity Distribution Bar Chart"""
    print("\n" + "="*60)
    print("GRAPH 1: ACTIVITY DISTRIBUTION BAR CHART")
    print("="*60)
    
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
    plt.show()
    print("✓ Activity Distribution Bar Chart displayed")

def show_graph_2_confidence_distribution(confidence_scores):
    """Show Graph 2: Confidence Score Distribution"""
    print("\n" + "="*60)
    print("GRAPH 2: CONFIDENCE SCORE DISTRIBUTION")
    print("="*60)
    
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
    plt.show()
    print("✓ Confidence Score Distribution displayed")

def show_graph_3_activity_timeline(predictions, timestamps, class_names):
    """Show Graph 3: Activity Timeline"""
    print("\n" + "="*60)
    print("GRAPH 3: ACTIVITY TIMELINE")
    print("="*60)
    
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
    plt.show()
    print("✓ Activity Timeline displayed")

def show_graph_4_activity_heatmap(predictions, class_names):
    """Show Graph 4: Activity Heatmap by Hour"""
    print("\n" + "="*60)
    print("GRAPH 4: ACTIVITY HEATMAP BY HOUR")
    print("="*60)
    
    plt.figure(figsize=(12, 6))
    
    # Create hourly activity matrix
    hours = 8
    activity_matrix = np.zeros((len(class_names), hours))
    
    # Simulate hourly distribution
    for i, pred in enumerate(predictions):
        hour = i % hours
        activity_matrix[pred, hour] += 1
    
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
    plt.show()
    print("✓ Activity Heatmap displayed")

def show_graph_5_per_class_confidence(predictions, confidence_scores, class_names):
    """Show Graph 5: Per-Class Confidence Analysis"""
    print("\n" + "="*60)
    print("GRAPH 5: PER-CLASS CONFIDENCE ANALYSIS")
    print("="*60)
    
    plt.figure(figsize=(12, 6))
    
    class_confidences = []
    class_counts = []
    
    for i, class_name in enumerate(class_names):
        class_mask = (predictions == i)
        if np.sum(class_mask) > 0:
            class_conf = np.mean(confidence_scores[class_mask])
            class_count = np.sum(class_mask)
        else:
            class_conf = 0
            class_count = 0
        
        class_confidences.append(class_conf)
        class_counts.append(class_count)
    
    colors = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462', '#B3DE69']
    bars = plt.bar(class_names, class_confidences, color=colors, alpha=0.7)
    
    plt.title('Average Confidence by Activity Class', fontsize=16, fontweight='bold')
    plt.xlabel('Activity Class', fontsize=12)
    plt.ylabel('Average Confidence', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, conf, count in zip(bars, class_confidences, class_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{conf:.3f}\n({count})', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    print("✓ Per-Class Confidence Analysis displayed")

def show_graph_6_activity_pie_chart(predictions, class_names):
    """Show Graph 6: Activity Distribution Pie Chart"""
    print("\n" + "="*60)
    print("GRAPH 6: ACTIVITY DISTRIBUTION PIE CHART")
    print("="*60)
    
    plt.figure(figsize=(10, 8))
    unique, counts = np.unique(predictions, return_counts=True)
    activity_counts = [counts[i] if i in unique else 0 for i in range(len(class_names))]
    
    colors = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462', '#B3DE69']
    
    # Only show slices with data
    non_zero_indices = [i for i, count in enumerate(activity_counts) if count > 0]
    non_zero_counts = [activity_counts[i] for i in non_zero_indices]
    non_zero_names = [class_names[i] for i in non_zero_indices]
    non_zero_colors = [colors[i] for i in non_zero_indices]
    
    wedges, texts, autotexts = plt.pie(non_zero_counts, labels=non_zero_names, colors=non_zero_colors,
                                      autopct='%1.1f%%', startangle=90)
    
    plt.title('Worker Activity Distribution (Pie Chart)', fontsize=16, fontweight='bold')
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    print("✓ Activity Distribution Pie Chart displayed")

def show_graph_7_confidence_vs_accuracy(predictions, confidence_scores):
    """Show Graph 7: Confidence vs Accuracy Analysis"""
    print("\n" + "="*60)
    print("GRAPH 7: CONFIDENCE VS ACCURACY ANALYSIS")
    print("="*60)
    
    plt.figure(figsize=(12, 6))
    
    # Create confidence bins
    confidence_bins = np.linspace(0, 1, 11)
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            # Simulate accuracy (higher confidence = higher accuracy)
            accuracy = 0.5 + 0.4 * bin_centers[i] + np.random.normal(0, 0.05)
            accuracy = max(0, min(1, accuracy))  # Clamp between 0 and 1
            accuracies.append(accuracy)
            bin_counts.append(np.sum(mask))
        else:
            accuracies.append(0)
            bin_counts.append(0)
    
    # Create scatter plot with size proportional to count
    sizes = [count * 10 for count in bin_counts]  # Scale for visibility
    plt.scatter(bin_centers, accuracies, s=sizes, alpha=0.6, c='red', edgecolors='black')
    
    # Add trend line
    z = np.polyfit(bin_centers, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(bin_centers, p(bin_centers), "r--", alpha=0.8, linewidth=2)
    
    plt.title('Confidence Score vs Model Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add annotations for some points
    for i, (x, y, count) in enumerate(zip(bin_centers, accuracies, bin_counts)):
        if count > 0 and i % 2 == 0:  # Annotate every other point
            plt.annotate(f'n={count}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    print("✓ Confidence vs Accuracy Analysis displayed")

def main():
    """Show all graphs exactly once in sequence."""
    print("="*80)
    print("WORKER ACTIVITY MONITORING - ALL GRAPHS DISPLAY")
    print("="*80)
    print("This script will show all 7 visualization graphs exactly once.")
    print("Close each graph window to proceed to the next one.")
    print("="*80)
    
    # Create sample data
    predictions, timestamps, confidence_scores = create_sample_data()
    class_names = ['sitting', 'standing', 'walking', 'working', 'resting', 'moving_objects', 'using_tools']
    
    print(f"\nData Summary:")
    print(f"- Total observations: {len(predictions)}")
    print(f"- Time period: 8 hours")
    print(f"- Activity classes: {len(class_names)}")
    print(f"- Average confidence: {np.mean(confidence_scores):.3f}")
    
    # Show all graphs in sequence
    show_graph_1_activity_distribution(predictions, class_names)
    show_graph_2_confidence_distribution(confidence_scores)
    show_graph_3_activity_timeline(predictions, timestamps, class_names)
    show_graph_4_activity_heatmap(predictions, class_names)
    show_graph_5_per_class_confidence(predictions, confidence_scores, class_names)
    show_graph_6_activity_pie_chart(predictions, class_names)
    show_graph_7_confidence_vs_accuracy(predictions, confidence_scores)
    
    print("\n" + "="*80)
    print("ALL GRAPHS DISPLAYED SUCCESSFULLY!")
    print("="*80)
    print("✓ Graph 1: Activity Distribution Bar Chart")
    print("✓ Graph 2: Confidence Score Distribution")
    print("✓ Graph 3: Activity Timeline")
    print("✓ Graph 4: Activity Heatmap by Hour")
    print("✓ Graph 5: Per-Class Confidence Analysis")
    print("✓ Graph 6: Activity Distribution Pie Chart")
    print("✓ Graph 7: Confidence vs Accuracy Analysis")
    print("="*80)
    print("All 7 graphs have been shown exactly once!")

if __name__ == "__main__":
    main()

