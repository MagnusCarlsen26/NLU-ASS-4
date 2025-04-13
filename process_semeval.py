import csv
import random
import os

# Read the dataset
data = []
try:
    with open('/home/khushal/Downloads/twitter.csv/training.1600000.processed.noemoticon.csv', 'r', encoding='latin1') as f:
        reader = csv.reader(f)
        for row in reader:
            # First column is sentiment (0=negative, 4=positive), last column is text
            sentiment = 'negative' if row[0] == '0' else 'positive'
            text = row[5]
            data.append((text, sentiment))
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Shuffle and split data
random.seed(42)
random.shuffle(data)

# Use only a subset for faster processing
sample_size = 10000
data = data[:sample_size]

# Split into train, dev, test (80%, 10%, 10%)
train_size = int(0.8 * len(data))
dev_size = int(0.1 * len(data))
train_data = data[:train_size]
dev_data = data[train_size:train_size+dev_size]
test_data = data[train_size+dev_size:]

# Create directory if it doesn't exist
os.makedirs('data/semeval_full', exist_ok=True)

# Write to files
header = 'ID\tTopic\tText\tSentiment'

try:
    with open('data/semeval_full/train.txt', 'w') as f:
        f.write(header + '\n')
        for i, (text, sentiment) in enumerate(train_data):
            f.write(f'{i+1}\tgeneral\t{text}\t{sentiment}\n')
            
    with open('data/semeval_full/dev.txt', 'w') as f:
        f.write(header + '\n')
        for i, (text, sentiment) in enumerate(dev_data):
            f.write(f'{i+1}\tgeneral\t{text}\t{sentiment}\n')
            
    with open('data/semeval_full/test.txt', 'w') as f:
        f.write(header + '\n')
        for i, (text, sentiment) in enumerate(test_data):
            f.write(f'{i+1}\tgeneral\t{text}\t{sentiment}\n')
            
    print('Files created successfully with sample size:', sample_size)
    print('Files located in data/semeval_full/')
except Exception as e:
    print(f"Error writing files: {e}")
    exit(1)

