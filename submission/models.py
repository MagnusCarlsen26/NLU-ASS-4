import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNN(nn.Module):
    """
    Feed-Forward Neural Network for sentiment analysis
    
    Architecture:
    - Embedding layer
    - Flatten
    - First hidden layer (size 256)
    - Second hidden layer (size 128)
    - Output layer (size depends on number of classes)
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_size1=256, hidden_size2=128, num_classes=2):
        super(FFNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Use a fixed size for simplicity - we'll use average pooling over sequence length
        self.fc1 = nn.Linear(embedding_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x has shape [batch_size, seq_len]
        
        # Get embeddings
        x = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Average pooling over sequence dimension
        x = torch.mean(x, dim=1)  # Shape: [batch_size, embedding_dim]
        
        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

class LSTM(nn.Module):
    """
    LSTM model for sentiment analysis
    
    Architecture:
    - Embedding layer
    - LSTM layer (hidden size 256)
    - Output layer (size depends on number of classes)
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=256, num_classes=2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x has shape [batch_size, seq_len]
        
        # Get embeddings
        x = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Run through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state for classification
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        
        return out 