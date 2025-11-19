from pathlib import Path
from dataset import CAMELOT_MAPPING
from tqdm import tqdm
import torchaudio
import librosa
import numpy as np
import pickle
import json
import argparse


def load_annotations_from_json(json_path, audio_base_dir):
    """
    Load annotations from correct_keys.json file.

    Args:
        json_path: Path to correct_keys.json
        audio_base_dir: Base directory containing audio subdirectories

    Returns:
        List of (audio_file_path, camelot_idx) tuples for valid high-confidence entries
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    data = []
    skipped_dual_key = 0
    skipped_low_confidence = 0
    missing_files = 0
    unknown_keys = []

    print(f"Loading annotations from: {json_path}")
    print(f"Total entries in JSON: {len(entries)}")

    for entry in entries:
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

        # Extract audio file path
        filename = entry.get('filename', '')
        audio_path = Path(audio_base_dir) / filename

        # Check if audio file exists
        if not audio_path.exists():
            missing_files += 1
            continue

        camelot_idx = CAMELOT_MAPPING[notation]
        data.append((audio_path, camelot_idx))

    # Print summary
    print(f"\nFiltering summary:")
    print(f"  High-confidence entries: {len(entries) - skipped_low_confidence}")
    print(f"  Skipped (low confidence): {skipped_low_confidence}")
    print(f"  Skipped (dual-key notation): {skipped_dual_key}")
    print(f"  Skipped (missing audio files): {missing_files}")
    print(f"  Valid entries: {len(data)}")

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

    return data


def preprocess_data(dataset_dir, output_dir, pitch_range = (-4, 7), json_path=None, audio_base_dir=None):
    """
    Preprocesses the MTG/GiantSteps Key Dataset for key classification, as in
    Korzeniowski & Widmer (2018).

    The function:
      - Loads all high-confidence audio files and corresponding key labels,
      - For each audio file, generates multiple pitch-shifted versions (data augmentation),
      - Computes the Constant-Q Transform (CQT) log-magnitude spectrogram with Librosa,
      - Stores results as .pkl files for efficient training use.

    Key differences to the original paper:
      - Uses librosa's `cqt` (Constant-Q Transform) extractor instead of approximation
        from Korzeniowski & Widmer (2017)
      - Parameterization of CQT (n_bins=105, bins_per_octave=24, fmin=65 Hz)

    Args:
        dataset_dir (Path): Path to original MTG dataset with 'audio' and 'annotations' folders (legacy mode).
        output_dir (Path): Target directory for preprocessed .pkl spectrogram files.
        pitch_range (): Augmentation range: semitone shifts
        json_path (Path): Optional path to correct_keys.json file
        audio_base_dir (Path): Optional base directory for audio files (when using JSON mode)
    """
    output_dir.mkdir(exist_ok=True)
    output_dir = Path(output_dir)
    sample_rate = 44100
    n_bins = 105             # Number of CQT bins: covers range with high resolution
    hop_length = 8820        # Large hop (approx. 0.2 sec at 44100 Hz, ~5 FPS) as in paper for global context

    # 1. Gather all high-confidence audio files and Camelot-encoded labels
    if json_path:
        # Load from JSON file
        print("Using JSON annotation file...")
        data = load_annotations_from_json(json_path, audio_base_dir)
    else:
        # Legacy mode: Load from annotations.txt
        print("Using legacy annotations.txt file...")
        audio_dir = Path(dataset_dir) / 'audio'
        annotations_path = Path(dataset_dir) / 'annotations' / 'annotations.txt'
        data = []

        with open(annotations_path, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    file_num, key_str, confidence = parts[0], parts[1], int(parts[2])
                    if key_str in CAMELOT_MAPPING and confidence == 2:
                        camelot_idx = CAMELOT_MAPPING[key_str]
                        filename = f'{file_num}.LOFI.mp3'
                        filepath = dataset_dir / 'audio' / filename
                        if filepath.exists():
                            data.append((filepath, camelot_idx))

    # 2. Iterate over files and create spectral representations for all pitch shifts
    print(f"\nPreprocessing {len(data)} audio files...")

    # Track failures
    failed_files = []
    failed_log_path = output_dir / 'failed_files.txt'

    # Check for resume capability
    processed_count = 0
    skipped_count = 0

    for filepath, _ in tqdm(data, desc="Processing"):
        # Extract base filename (without extension)
        base_filename = filepath.stem

        # Check if all pitch-shifted versions already exist (resume capability)
        expected_files = [output_dir / f'{base_filename}_{n}.pkl'
                         for n in range(pitch_range[0], pitch_range[1] + 1)]

        if all(f.exists() for f in expected_files):
            skipped_count += 1
            continue

        try:
            # Load audio file (using librosa for better compatibility across formats)
            waveform, sr = librosa.load(filepath, sr=sample_rate, mono=True)
            waveform = waveform.astype(np.float32)

            # For each pitch shift in the augmentation window
            for n_steps in range(pitch_range[0], pitch_range[1] + 1):
                out_file = output_dir / f'{base_filename}_{n_steps}.pkl'
                if out_file.exists():
                    continue
                # Apply pitch shift except when n_steps==0 (original)
                if n_steps != 0:
                    shifted_waveform = librosa.effects.pitch_shift(waveform.astype(np.float32), sr=sample_rate, n_steps=n_steps)
                else:
                    shifted_waveform = waveform
                # Compute CQT (log-frequency spectrogram), following the paper's input representation
                cqt = librosa.cqt(
                    shifted_waveform,
                    sr=sample_rate,
                    hop_length=hop_length,
                    n_bins=n_bins,
                    bins_per_octave=24,
                    fmin=65,                       # Lowest frequency bin (Hz)
                )
                spec = np.abs(cqt)                 # Only magnitude is used
                spec = np.log1p(spec)              # Log-magnitude for dynamic range compression, as in the paper
                # Save as pickled numpy array for later efficient loading during training
                with open(out_file, 'wb') as f:
                    pickle.dump(spec, f)

            processed_count += 1

        except Exception as e:
            error_msg = f"{filepath.name}: {type(e).__name__}: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            failed_files.append((str(filepath), str(e)))

            # Log failure immediately
            with open(failed_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{filepath}\t{type(e).__name__}: {str(e)}\n")

            continue  # Skip to next file

    # Print summary
    print(f"\n{'='*80}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total files:          {len(data)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"Failed:                {len(failed_files)}")
    print()

    if failed_files:
        print(f"Failed files logged to: {failed_log_path}")
        print(f"\nFirst {min(10, len(failed_files))} failures:")
        for filepath, error in failed_files[:10]:
            print(f"  {Path(filepath).name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

def create_annotations_txt(dataset_dir):
    """
    Creates an 'annotations.txt' file for the GiantSteps dataset in the same
    format as the MTG (GiantSteps-MTG) dataset, enabling unified preprocessing 
    and data loading.

    For GiantSteps, original labels are in individual .key files (per track).
    Unlike MTG, there is no explicit confidence indicator.
    To mimic MTG format, all entries are assigned high confidence (2).

    The generated file will have one header line and then tab-separated lines:
      ID    MANUAL KEY    C
    where C = 2 for all entries (high confidence).

    Args:
        dataset_dir (str or Path): Path to GiantSteps dataset root (must contain
                                   'annotations/giantsteps/' with .key files).
    """
    dataset_dir = Path(dataset_dir)
    # Find all .key files in the standard GiantSteps annotation folder
    files = (dataset_dir / 'annotations' / 'giantsteps').glob("*.key")

    data = []

    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
            # Only process files with exactly two lines (ID + key information)
            if len(lines) != 2:
                continue
            l = lines[1]
            parts = l.strip().split(" ")
            # The second line is expected to contain key information as four space-separated fields
            if len(parts) != 4:
                print(parts)
                continue
            key_str = f'{parts[2]} {parts[3]}'
            # The file number (ID) is extracted from the file name by stripping "_LOFI" and extension
            file_num = file.stem[:-5]
            data.append([file_num, key_str])

    # Write the unified annotations.txt file, matching the MTG format for workflow compatibility
    with open(dataset_dir / 'annotations' / 'annotations.txt', 'w') as writer:
        writer.writelines(['ID\tMANUAL KEY\tC\n'])
        for d in data:
            # Each line: file_num, key_str, confidence=2 ('high')
            writer.writelines([f'{d[0]}\t{d[1]}\t2\n'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess audio files for key detection training')
    parser.add_argument('--json-file', type=str, help='Path to correct_keys.json file')
    parser.add_argument('--audio-base-dir', type=str, help='Base directory containing audio subdirectories (required with --json-file)')
    parser.add_argument('--output-dir', type=str, default='Dataset/preprocessed', help='Output directory for preprocessed spectrograms')
    parser.add_argument('--dataset-dir', type=str, help='Dataset directory with audio and annotations folders (legacy mode)')
    parser.add_argument('--pitch-range-min', type=int, default=-4, help='Minimum pitch shift in semitones')
    parser.add_argument('--pitch-range-max', type=int, default=7, help='Maximum pitch shift in semitones')

    args = parser.parse_args()

    # Validate arguments
    if args.json_file:
        if not args.audio_base_dir:
            parser.error('--audio-base-dir is required when using --json-file')

        print("=== JSON Mode ===")
        preprocess_data(
            dataset_dir=None,
            output_dir=Path(args.output_dir),
            pitch_range=(args.pitch_range_min, args.pitch_range_max),
            json_path=Path(args.json_file),
            audio_base_dir=Path(args.audio_base_dir)
        )
    else:
        if not args.dataset_dir:
            # Default legacy mode
            print("=== Legacy Mode (default datasets) ===")
            # --- This part is needed for training ---
            dataset_dir = Path('Dataset') / 'giantsteps-mtg-key-dataset'
            output_dir = Path('Dataset') / 'mtg-preprocessed-audio'
            preprocess_data(dataset_dir, output_dir)
            # --- This part is needed for evaluation ---
            dataset_dir = Path('Dataset') / 'giantsteps-key-dataset'
            output_dir = Path('Dataset') / 'giantsteps-preprocessed-audio'
            create_annotations_txt(dataset_dir)
            preprocess_data(dataset_dir, output_dir, pitch_range=(0,0))
        else:
            print("=== Legacy Mode (custom dataset) ===")
            preprocess_data(
                dataset_dir=Path(args.dataset_dir),
                output_dir=Path(args.output_dir),
                pitch_range=(args.pitch_range_min, args.pitch_range_max)
            )