import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse

from model import KeyNet
from dataset import KeyDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train key detection model')
    parser.add_argument('--json-file', type=str, help='Path to correct_keys.json file')
    parser.add_argument('--preprocessed-dir', type=str, default='Dataset/mtg-preprocessed-audio', help='Directory with preprocessed spectrograms')
    parser.add_argument('--dataset-dir', type=str, help='Dataset directory (legacy mode)')
    parser.add_argument('--model-name', type=str, help='Output model filename (saved in checkpoints/)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--num-epochs', type=int, default=2000, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--pitch-range-min', type=int, default=-4, help='Minimum pitch shift')
    parser.add_argument('--pitch-range-max', type=int, default=7, help='Maximum pitch shift')

    args = parser.parse_args()

    # File where the best model weights will be stored
    model_file_path = Path('checkpoints') / args.model_name
    model_file_path.parent.mkdir(exist_ok=True)

    # --- Hyperparameters ---
    # Check for MPS (Apple Silicon), CUDA (NVIDIA), or fall back to CPU
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    BATCH_SIZE      = args.batch_size
    LEARNING_RATE   = args.learning_rate
    NUM_EPOCHS      = args.num_epochs
    PATIENCE        = args.patience
    pitch_range     = (args.pitch_range_min, args.pitch_range_max)

    # 1. Load and split the dataset
    if args.json_file:
        print(f"=== Training with JSON annotations ===")
        print(f"JSON file: {args.json_file}")
        print(f"Preprocessed dir: {args.preprocessed_dir}")
        dataset = KeyDataset(
            root_dir=None,
            preprocessed_dir=Path(args.preprocessed_dir),
            pitch_range=pitch_range,
            json_path=Path(args.json_file)
        )
    else:
        print(f"=== Training with legacy annotations.txt ===")
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path('Dataset') / 'giantsteps-mtg-key-dataset'
        preprocessed_dir = Path(args.preprocessed_dir)
        print(f"Dataset dir: {dataset_dir}")
        print(f"Preprocessed dir: {preprocessed_dir}")
        dataset = KeyDataset(dataset_dir, preprocessed_dir, pitch_range=pitch_range)

    # Use a standard 80/20 train/validation split
    print(f"\nDataset size: {len(dataset)}")
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    print(f"Train set: {train_len}, Validation set: {val_len}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Initialize model, criterion, optimizer
    print(f"\nModel output: {model_file_path}")
    print(f"Device: {DEVICE}")
    model = KeyNet(num_classes=24, in_channels=1, Nf=64, p=0.3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training/Validation loop with early stopping and learning rate scheduling
    print(f"\n=== Starting Training ===")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Patience: {PATIENCE}\n")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # --- Training phase ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            inputs = batch['spec'].to(DEVICE)      # Spectrogram chunks
            labels = batch['gt_id'].to(DEVICE)     # Camelot-mapped key indices

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        avg_train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation phase ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['spec'].to(DEVICE)
                labels = batch['gt_id'].to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / total
        val_acc = correct / total

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
              f"TrainLoss: {avg_train_loss:.4f} | TrainAcc: {train_acc:.4f} | "
              f"ValLoss: {avg_val_loss:.4f} | ValAcc: {val_acc:.4f}"
        )

        # --- Early stopping and learning rate adjustment ---
        # Save best model and reset patience if validation improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_file_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                model.load_state_dict(torch.load(model_file_path))
                patience_counter = 0
                LEARNING_RATE /= 2
                print("Learning rate halved.")
                # Update optimizer with new learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LEARNING_RATE
            # Stop if learning rate is too small; this condition prevents endless fine-tuning
            if LEARNING_RATE < 1e-7:
                print("Early stopping: minimal learning rate reached.")
                break

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_file_path}")
