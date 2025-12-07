import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px

def save_model_pipeline(model, preprocessor, filepath='models/pneumonia_model.pkl'):
    """Save the complete model pipeline using joblib"""
    pipeline = {
        'model': model,
        'img_size': preprocessor.img_size,
        'class_names': preprocessor.class_names,
        'version': '1.0.0'
    }
    
    joblib.dump(pipeline, filepath)
    print(f"Model saved to {filepath}")

def load_model_pipeline(filepath='models/pneumonia_model.pkl'):
    """Load the model pipeline"""
    pipeline = joblib.load(filepath)
    return pipeline

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        ax.plot(history.history[metric], label=f'Training {metric}')
        ax.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        ax.set_title(f'Model {metric}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix_plotly(y_true, y_pred, class_names):
    """Create interactive confusion matrix using Plotly"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=500,
        height=500
    )
    
    return fig

def calculate_class_weights(train_generator):
    """Calculate class weights for imbalanced dataset"""
    class_counts = train_generator.classes
    total_samples = len(class_counts)
    unique_classes = np.unique(class_counts)
    
    class_weights = {}
    for cls in unique_classes:
        class_weights[cls] = total_samples / (len(unique_classes) * np.sum(class_counts == cls))
    
    return class_weights