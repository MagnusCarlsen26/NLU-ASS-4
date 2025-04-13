import os
import subprocess
import argparse

def run_experiment(model, dataset, data_dir, batch_size=32, epochs=2, lr=0.001):
    """
    Run a single experiment
    """
    cmd = [
        "python", "train.py",
        "--model", model,
        "--dataset", dataset,
        "--data_dir", data_dir,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_all_experiments(imdb_dir, semeval_dir, epochs=2):
    """
    Run all experiments (FFNN and LSTM on both datasets)
    """
    # FFNN on IMDB
    run_experiment("ffnn", "imdb", imdb_dir, epochs=epochs)
    
    # LSTM on IMDB
    run_experiment("lstm", "imdb", imdb_dir, epochs=epochs)
    
    # FFNN on SemEval
    run_experiment("ffnn", "semeval", semeval_dir, epochs=epochs)
    
    # LSTM on SemEval
    run_experiment("lstm", "semeval", semeval_dir, epochs=epochs)

def main():
    parser = argparse.ArgumentParser(description='Run sentiment analysis experiments')
    parser.add_argument('--mode', choices=['all', 'single'], default='all', help='Run all experiments or a single one')
    parser.add_argument('--model', choices=['ffnn', 'lstm'], help='Model to use (required if mode is single)')
    parser.add_argument('--dataset', choices=['imdb', 'semeval'], help='Dataset to use (required if mode is single)')
    parser.add_argument('--imdb_dir', required=True, help='Directory containing the IMDB dataset')
    parser.add_argument('--semeval_dir', required=True, help='Directory containing the SemEval dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        run_all_experiments(args.imdb_dir, args.semeval_dir, epochs=args.epochs)
    else:  # mode == 'single'
        if not args.model or not args.dataset:
            parser.error("--model and --dataset are required when mode is 'single'")
        
        data_dir = args.imdb_dir if args.dataset == 'imdb' else args.semeval_dir
        run_experiment(args.model, args.dataset, data_dir, args.batch_size, args.epochs, args.lr)

if __name__ == '__main__':
    main() 