from pathlib import Path
import random
import torch
import pickle
import json
from torch.utils.data import Dataset
from tqdm import tqdm

# Mapping from key string to Camelot Wheel index
# (1A = 0, ..., 12A = 11, 1B = 12, ..., 12B = 23)
CAMELOT_MAPPING = {
    'G# minor': 0,
    'Ab minor': 0,
    'D# minor': 1,
    'Eb minor': 1,
    'A# minor': 2,
    'Bb minor': 2,
    'F minor': 3,
    'C minor': 4,
    'G minor': 5,
    'D minor': 6,
    'A minor': 7,
    'E minor': 8,
    'B minor': 9,
    'F# minor': 10,
    'Gb minor': 10,
    'C# minor': 11,
    'Db minor': 11,
    'B major': 12,
    'F# major': 13,
    'Gb major': 13,
    'C# major': 14,
    'Db major': 14,
    'G# major': 15,
    'Ab major': 15,
    'D# major': 16,
    'Eb major': 16,
    'A# major': 17,
    'Bb major': 17,
    'F major': 18,
    'C major': 19,
    'G major': 20,
    'D major': 21,
    'A major': 22,
    'E major': 23
}

class KeyDataset(Dataset):
    """
    Dataset class for key classification, following Korzeniowski & Widmer (2018). It uses precomputed
    log-frequency spectrograms and pitch-shifting augmentation to improve key-robustness.

    Each entry returns a spectrogram chunk and a ground-truth key label.
    Key labels are mapped using the Camelot Wheel convention, which encodes enharmonic equivalents.

    Args:
        root_dir (str or Path): Root directory containing the official dataset and annotations.
        preprocessed_dir (str or Path): Directory with precomputed and pitch-shifted spectrograms (.pkl files).
        chunk_samples (int): Number of time frames in each spectrogram snippet (default: 100, ~20s).
        pitch_range (tuple): Min and max (inclusive) semitone steps for data augmentation.

    Attributes:
        data (list): List of (filename, camelot_index) pairs for valid preprocessed data.
    """
    def __init__(self, root_dir, preprocessed_dir, chunk_samples=100, pitch_range=(-4, 7), json_path=None):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.chunk_samples = chunk_samples
        self.pitch_range = pitch_range

        self.data = []

        if json_path:
            # Load from JSON file
            self._load_from_json(json_path)
        else:
            # Legacy mode: load from annotations.txt
            root = Path(root_dir)
            self.annotations_path = root / 'annotations' / 'annotations.txt'

            # Pre-cache all spectrogram files to avoid repeated glob calls
            print("Caching spectrogram file list...")
            all_spectrogram_files = set(f.stem for f in self.preprocessed_dir.glob('*.pkl'))
            print(f"Found {len(all_spectrogram_files)} spectrogram files")

            with open(self.annotations_path, "r") as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 3:
                        file_num, key_str, confidence = parts[0], parts[1], int(parts[2])
                        # Ensure the key is in the mapping and has high confidence
                        if key_str in CAMELOT_MAPPING and confidence == 2:
                            camelot_idx = CAMELOT_MAPPING[key_str]
                            filename = f"{file_num}.LOFI"
                            # Check that all expected pitch-shifted spectrograms exist using cached set
                            expected = self.pitch_range[1] - self.pitch_range[0] + 1
                            files_found = sum(1 for shift in range(self.pitch_range[0], self.pitch_range[1] + 1)
                                              if f'{filename}_{shift}' in all_spectrogram_files)
                            if files_found < expected:
                                print(f'File {filename} not preprocessed correctly. Found {files_found} spectrograms.')
                                continue
                            self.data.append((filename, camelot_idx))

    def _load_from_json(self, json_path):
        """
        Load dataset from correct_keys.json file.

        Args:
            json_path: Path to correct_keys.json file
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)

        skipped_dual_key = 0
        skipped_low_confidence = 0
        missing_spectrograms = 0
        unknown_keys = []

        print(f"Loading dataset from JSON: {json_path}")
        print(f"Total entries in JSON: {len(entries)}")

        # Pre-cache all spectrogram files to avoid repeated glob calls
        # This prevents "too many open files" errors during long training runs
        print("Caching spectrogram file list...")
        all_spectrogram_files = set(f.stem for f in self.preprocessed_dir.glob('*.pkl'))
        print(f"Found {len(all_spectrogram_files)} spectrogram files")

        for entry in tqdm(entries, desc="Validating entries", unit="entry"):
            # Filter by confidence
            if entry.get('confidence') != 'high':
                skipped_low_confidence += 1
                continue

            notation = entry.get('notation', '')

            # Skip dual-key entries (contain "/")
            if '/' in notation:
                skipped_dual_key += 1
                continue

            # Validate notation exists in CAMELOT_MAPPING
            if notation not in CAMELOT_MAPPING:
                unknown_keys.append((entry.get('filename', 'unknown'), notation))
                continue

            camelot_idx = CAMELOT_MAPPING[notation]

            # Extract base filename (remove folder prefix and extension)
            filename_full = entry.get('filename', '')
            # Extract just the filename part (e.g., "dataset mtg/5061.LOFI.mp3" -> "5061.LOFI")
            filename = Path(filename_full).stem

            # Check that all expected pitch-shifted spectrograms exist using cached set
            expected = self.pitch_range[1] - self.pitch_range[0] + 1
            files_found = sum(1 for shift in range(self.pitch_range[0], self.pitch_range[1] + 1)
                              if f'{filename}_{shift}' in all_spectrogram_files)
            if files_found < expected:
                missing_spectrograms += 1
                continue

            self.data.append((filename, camelot_idx))

        # Print summary
        print(f"\nDataset loading summary:")
        print(f"  High-confidence entries: {len(entries) - skipped_low_confidence}")
        print(f"  Skipped (low confidence): {skipped_low_confidence}")
        print(f"  Skipped (dual-key notation): {skipped_dual_key}")
        print(f"  Skipped (missing spectrograms): {missing_spectrograms}")
        print(f"  Valid entries loaded: {len(self.data)}")

        # Error out if unknown keys found
        if unknown_keys:
            print(f"\nERROR: Found {len(unknown_keys)} entries with unknown key notation:")
            for filename, notation in unknown_keys[:10]:
                print(f"  - {filename}: '{notation}'")
            if len(unknown_keys) > 10:
                print(f"  ... and {len(unknown_keys) - 10} more")
            print(f"\nValid keys are:")
            for key in sorted(CAMELOT_MAPPING.keys()):
                print(f"  - {key}")
            raise ValueError(f"Found {len(unknown_keys)} entries with unknown key notation. Please fix the JSON file.")

    def __len__(self):
        """
        Returns:
            int: Total number of song/key pairs with high confidence in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a randomly pitch-shifted spectrogram chunk and the corresponding Camelot key ID.

        1. Randomly selects a pitch shift within the defined range.
        2. Adjusts the Camelot key index to preserve key mode after shift.
        3. Loads the corresponding preprocessed spectrogram file.
        4. Randomly extracts a chunk of given length.

        Args:
            idx (int): Index in the dataset.

        Returns:
            dict: {'spec': torch.Tensor [C, F, T], 'gt_id': int}
        """
        filename, camelot_idx = self.data[idx]

        n_steps = random.randint(self.pitch_range[0], self.pitch_range[1])
        # Compute Camelot key shift according to wheel, keeping minor/major mode fixed
        if n_steps % 2 == 0:
            camelot_steps = n_steps
        else:
            # For odd steps, correction by wheel symmetry to preserve relative key position
            camelot_steps = n_steps + 6
        if n_steps != 0:
            if camelot_idx < 12:  # minor key
                camelot_idx = (camelot_idx + camelot_steps) % 12
            else:  # major key
                camelot_idx = (camelot_idx - 12 + camelot_steps) % 12 + 12

        # Load pitch-shifted spectrogram from preprocessed data
        with open(self.preprocessed_dir / f'{filename}_{n_steps}.pkl', 'rb') as f:
            full_spec = pickle.load(f)

        # Extract a chunk of specified length at a random time offset
        max_start = max(len(full_spec[1]) - self.chunk_samples - 1,0)
        start_sample = random.randint(0, max_start)
        length = min(self.chunk_samples, len(full_spec[1]) - 1)  # Ignore last bin
        chunk = full_spec[:, start_sample:start_sample+length]

        # Ensure output has correct shape: (channel, freq, time)
        spec = torch.tensor(chunk, dtype=torch.float32)
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)   # (1, freq, time)

        return {'spec': spec, 'gt_id': camelot_idx}

if __name__ == '__main__':
    # Example usage: Load data and print sample shape
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset_dir = Path('Dataset') / 'giantsteps-mtg-key-dataset'
    preprocessed_dir = Path('Dataset') / 'mtg-preprocessed-audio'

    dataset = KeyDataset(dataset_dir, preprocessed_dir)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f'Dataset size: {len(dataset)}')

    for sample in tqdm(train_loader):
        spec = sample['spec']
        gt_id = sample['gt_id']
        print('Spec shape:', spec.shape)
        break
