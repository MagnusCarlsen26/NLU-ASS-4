import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from utils import (
    load_imdb_dataset, 
    load_semeval_dataset, 
    Vocabulary, 
    calculate_average_length, 
    prepare_dataloaders,
    SentimentDataset
)
from models import FFNN, LSTM

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs=10, model_name="model"):
    """
    Train the model and save checkpoints
    """
    # Track metrics
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_saved = False
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Training loop
        for inputs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        
        # Validation
        val_acc = evaluate(model, val_dataloader, device)
        
        # Handle NaN values
        if np.isnan(val_acc):
            val_acc = 0.0
        
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            best_model_saved = True
    
    # If no model was saved due to all NaN accuracies, save the last model
    if not best_model_saved:
        torch.save(model.state_dict(), f'{model_name}_best.pth')
        print("All validation accuracies were NaN. Saving the last model as the best model.")
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training.png')
    
    return train_losses, val_accuracies

def evaluate(model, dataloader, device):
    """
    Evaluate the model on a dataloader
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    if len(all_labels) > 0 and len(all_preds) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
    else:
        accuracy = float('nan')
    
    return accuracy

def test_model(model, test_dataloader, device, class_names):
    """
    Test the model and report metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    if len(all_labels) == 0 or len(all_preds) == 0:
        print("No data to evaluate")
        return 0, 0, 0, 0, {}
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Class-wise metrics
    try:
        class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    except Exception as e:
        print(f"Error computing classification report: {e}")
        class_report = {name: {"precision": 0, "recall": 0, "f1-score": 0} for name in class_names}
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    
    # Print class-wise metrics
    print("\nClass-wise metrics:")
    for class_name in class_names:
        if class_name in class_report:
            print(f"{class_name}:")
            print(f"  Precision: {class_report[class_name]['precision']:.4f}")
            print(f"  Recall: {class_report[class_name]['recall']:.4f}")
            print(f"  F1-Score: {class_report[class_name]['f1-score']:.4f}")
    
    return accuracy, precision, recall, f1, class_report

def main():
    parser = argparse.ArgumentParser(description='Train models for sentiment analysis')
    parser.add_argument('--dataset', type=str, choices=['imdb', 'semeval'], required=True, help='Dataset to use')
    parser.add_argument('--model', type=str, choices=['ffnn', 'lstm'], required=True, help='Model to train')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    if args.dataset == 'imdb':
        train_dir = os.path.join(args.data_dir, 'train')
        test_dir = os.path.join(args.data_dir, 'test')
        
        train_texts, train_labels = load_imdb_dataset(train_dir)
        test_texts, test_labels = load_imdb_dataset(test_dir)
        
        # Split train set to get validation set (10% of train set)
        val_size = max(1, int(0.1 * len(train_texts)))
        train_size = len(train_texts) - val_size
        
        # Ensure train_size is at least 1
        train_size = max(1, train_size)
        
        train_texts, val_texts = train_texts[:train_size], train_texts[train_size:]
        train_labels, val_labels = train_labels[:train_size], train_labels[train_size:]
        
        num_classes = 2
        class_names = ['negative', 'positive']
        
    elif args.dataset == 'semeval':
        train_file = os.path.join(args.data_dir, 'train.txt')
        val_file = os.path.join(args.data_dir, 'dev.txt')
        test_file = os.path.join(args.data_dir, 'test.txt')
        
        train_texts, train_labels = load_semeval_dataset(train_file)
        val_texts, val_labels = load_semeval_dataset(val_file)
        test_texts, test_labels = load_semeval_dataset(test_file)
        
        num_classes = 3
        class_names = ['negative', 'neutral', 'positive']
    
    # Build vocabulary
    vocab = Vocabulary(min_freq=1)  # Changed from 5 to 1 for small test dataset
    for text in train_texts:
        vocab.add_sentence(text)
    vocab.build_vocab()
    
    # Calculate average length for padding/truncation
    avg_length = max(1, calculate_average_length(train_texts))
    print(f"Average length of texts: {avg_length}")
    
    # Prepare dataloaders
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, avg_length)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab, avg_length)
    test_dataset = SentimentDataset(test_texts, test_labels, vocab, avg_length)
    
    # Create dataloaders with safe batch sizes
    train_dataloader = DataLoader(train_dataset, batch_size=max(1, min(args.batch_size, len(train_dataset))), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=max(1, min(args.batch_size, len(val_dataset))), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=max(1, min(args.batch_size, len(test_dataset))), shuffle=False)
    
    # Initialize model based on model type
    if args.model == 'ffnn':
        model_name = f'ffnn_{args.dataset}'
        vocab_size = vocab.n_words
        model = FFNN(vocab_size=vocab_size, num_classes=num_classes).to(device)
        print(f"Training ffnn on {args.dataset} dataset...")
    else:  # lstm
        model_name = f'lstm_{args.dataset}'
        vocab_size = vocab.n_words
        model = LSTM(vocab_size=vocab_size, num_classes=num_classes).to(device)
        print(f"Training lstm on {args.dataset} dataset...")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    train_losses, val_accuracies = train_model(
        model, train_dataloader, val_dataloader, criterion, optimizer, device, 
        epochs=args.epochs, model_name=model_name
    )
    
    # Load best model if it exists
    best_model_path = f'{model_name}_best.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    # Test model
    print(f"Testing {args.model} on {args.dataset} dataset...")
    accuracy, precision, recall, f1, class_report = test_model(model, test_dataloader, device, class_names)
    
    # Save metrics to a file
    with open(f'{model_name}_metrics.txt', 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Test F1-Score: {f1:.4f}\n\n")
        
        f.write("Class-wise metrics:\n")
        for class_name in class_names:
            if class_name in class_report:
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {class_report[class_name]['precision']:.4f}\n")
                f.write(f"  Recall: {class_report[class_name]['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_report[class_name]['f1-score']:.4f}\n\n")

if __name__ == '__main__':
    main() 