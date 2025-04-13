import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Directory where metrics files are stored
model_metrics_dir = '.'

# Function to extract metrics from a file
def extract_metrics_from_file(file_path):
    epochs = []
    train_losses = []
    val_accuracies = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # Look for lines with epoch information
            epoch_match = re.search(r'Epoch (\d+)/\d+, Loss: ([\d\.]+), Val Accuracy: ([\d\.]+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                train_loss = float(epoch_match.group(2))
                val_accuracy = float(epoch_match.group(3))
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_accuracies.append(val_accuracy)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return epochs, train_losses, val_accuracies

# Function to plot metrics for a specific model and dataset
def plot_metrics(model_name, dataset_name):
    log_file = f"{model_name}_{dataset_name}_metrics.txt"
    log_path = os.path.join(model_metrics_dir, log_file)
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    
    epochs, train_losses, val_accuracies = extract_metrics_from_file(log_path)
    
    if not epochs:
        print(f"No metrics found in {log_path}")
        return
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot validation accuracy
    ax1.plot(epochs, val_accuracies, 'b-o', label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{model_name.upper()} on {dataset_name.upper()}: Validation Accuracy')
    ax1.grid(True)
    
    # Plot training loss
    ax2.plot(epochs, train_losses, 'r-o', label='Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{model_name.upper()} on {dataset_name.upper()}: Training Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{model_name}_{dataset_name}_training_metrics.png")
    print(f"Metrics plotted and saved as {model_name}_{dataset_name}_training_metrics.png")

# Function to plot all combinations of models and datasets
def plot_all_metrics():
    models = ['ffnn', 'lstm']
    datasets = ['imdb', 'semeval']
    
    for model in models:
        for dataset in datasets:
            print(f"Processing {model} on {dataset}...")
            plot_metrics(model, dataset)

if __name__ == "__main__":
    print("Plotting training metrics...")
    
    # Create directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Find all metrics files in the current directory
    metrics_files = [f for f in os.listdir(model_metrics_dir) if f.endswith('_metrics.txt')]
    
    if metrics_files:
        print(f"Found metrics files: {metrics_files}")
        plot_all_metrics()
    else:
        print("No metrics files found in the current directory.")
        print("Please make sure that metrics files are named in the format: {model}_{dataset}_metrics.txt") 