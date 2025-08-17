"""Professional model performance visualization."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class ModelPerformancePlotter:
    """Professional plotting utilities for model performance analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: tuple = (10, 6)):
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
        
    def plot_training_comparison(self, models: List[str], times: List[float], 
                               title: str = "Model Training Time Comparison") -> None:
        """Plot training time comparison between models."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.bar(models, times, color=self.colors[:len(models)])
        
        # Customize plot
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Training Time (minutes)', fontsize=12)
        ax.set_xlabel('Models', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}m',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontweight='bold')
        
        # Styling
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
    def plot_performance_metrics(self, performance_data: Dict[str, Dict]) -> None:
        """Plot comprehensive performance metrics comparison."""
        models = list(performance_data.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [performance_data[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, values, color=self.colors[:len(models)])
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_ylim(0, 1)
            axes[i].set_ylabel('Score')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[i].annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom')
            
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def plot_model_size_vs_performance(self, model_data: List[Dict]) -> None:
        """Plot model size vs performance trade-off."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sizes = [d['size_mb'] for d in model_data]
        f1_scores = [d['f1_score'] for d in model_data]
        names = [d['name'] for d in model_data]
        
        scatter = ax.scatter(sizes, f1_scores, s=100, c=self.colors[:len(model_data)], alpha=0.7)
        
        # Add model names as labels
        for i, name in enumerate(names):
            ax.annotate(name, (sizes[i], f1_scores[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Model Size (MB)', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Model Size vs Performance Trade-off', fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
    def save_plot(self, filename: str, output_dir: str = "reports/plots") -> None:
        """Save the current plot to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path / filename}")