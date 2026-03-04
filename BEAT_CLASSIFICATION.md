# First Beat Detection via CNN Classification

This document outlines an approach to detect the first beat (downbeat) of a track using CNN-based classification, similar to the key detection approach in this project.

---

## Problem Overview

### Current Approach (lexicon-beatdetector)
- Uses traditional DSP: FFT, spectral flux, correlation, grid-walking
- Works well for BPM detection
- First beat detection is less reliable, especially for tracks with intros

### Proposed Approach
- Use a CNN trained on labeled data to classify beat phase
- Leverage known accurate BPM to simplify the problem
- Frame as classification rather than regression

---

## Problem Framing

### Why Classification Works

Since BPM is already known accurately, we don't need to detect tempo. The problem becomes:

> "Given a track with known BPM, which beat position aligns with the musical downbeat?"

This is analogous to key detection:
- **Key detection**: "Which of 24 keys is this track in?"
- **Beat phase detection**: "Which of N phase positions is the first beat at?"

### Phase Quantization

At a given BPM, beats occur at regular intervals. For 120 BPM:
- Beat interval = 60/120 = 0.5 seconds
- Bar interval (4/4 time) = 2.0 seconds

The first beat must fall at one of these quantized positions (or close to it). We can classify:

1. **Beat phase within bar** (4 classes for 4/4): Which beat of the bar does position 0 align with?
2. **Bar offset** (N classes): How many bars into the track does the music "start"?

### Simplified Model: Phase-Only Classification

For the simplest approach, assume:
- All music is 4/4 time
- We analyze a window that contains the first musical downbeat
- Output: phase offset (0, 1, 2, or 3 beats)

```
Phase 0: First beat at 0.000s, 2.000s, 4.000s... (at 120 BPM)
Phase 1: First beat at 0.500s, 2.500s, 4.500s...
Phase 2: First beat at 1.000s, 3.000s, 5.000s...
Phase 3: First beat at 1.500s, 3.500s, 5.500s...
```

---

## Architecture Options

### Option A: Global Classification (KeyNet-style)

Similar to key detection — analyze a chunk and output a single class.

```
Input: Mel/CQT spectrogram of first 30 seconds
       Shape: (1, freq_bins, time_frames)

Network: CNN with global average pooling
         9 conv layers → global pool → FC → 4 classes

Output: Softmax over 4 phase classes
```

**Pros:**
- Simple, proven architecture
- Easy to train with existing pipeline

**Cons:**
- Loses temporal precision
- Doesn't handle variable intro lengths well

### Option B: Per-Frame Classification

Output a probability for each time frame, then post-process.

```
Input: Mel spectrogram of first 30 seconds
       Shape: (1, freq_bins, time_frames)

Network: Fully convolutional (no global pooling)
         Conv layers → 1x1 conv → sigmoid

Output: Per-frame downbeat probability
        Shape: (time_frames,)
```

**Pros:**
- Temporal precision
- Can handle variable intros
- Interpretable output (probability curve)

**Cons:**
- More complex training (need frame-level labels)
- Post-processing required

### Option C: Hybrid — Segment Classification

Divide track into beat-aligned segments, classify each.

```
Input: Multiple spectrograms, one per potential beat position
       Shape: (num_candidates, 1, freq_bins, segment_frames)

Network: Siamese-style CNN
         Shared weights process each candidate

Output: Score for each candidate being the downbeat
```

**Pros:**
- Explicit candidate comparison
- Works well with known BPM

**Cons:**
- More complex data pipeline
- Slower inference

### Recommended: Option B (Per-Frame) with Post-Processing

This offers the best balance of precision and flexibility.

---

## Input Features

### Spectrogram Choice

| Feature | Key Detection | Beat Detection |
|---------|--------------|----------------|
| **CQT** | ✓ (harmonic content) | Possible but not ideal |
| **Mel spectrogram** | Less common | ✓ (better for percussive) |
| **Onset strength** | Not used | ✓ (rhythm-focused) |
| **Multi-feature** | Not needed | Could combine mel + onset |

### Recommended: Mel Spectrogram + Onset Strength

```python
# Mel spectrogram (captures timbral/harmonic content)
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=44100,
    n_mels=80,
    fmin=20,
    fmax=8000,
    hop_length=512  # ~11.6ms resolution
)
mel_db = librosa.power_to_db(mel, ref=np.max)

# Onset strength envelope (captures rhythmic attacks)
onset_env = librosa.onset.onset_strength(
    y=audio,
    sr=44100,
    hop_length=512
)

# Stack as 2-channel input
# Shape: (2, 80, time_frames) — mel + onset as channels
```

### Alternative: Rhythm-Focused Features

```python
# Tempogram (tempo-centric representation)
tempogram = librosa.feature.tempogram(
    onset_envelope=onset_env,
    sr=44100,
    hop_length=512
)

# Beat-synchronous features
# Aggregate features at beat positions (using known BPM)
beat_frames = librosa.time_to_frames(beat_times, sr=44100, hop_length=512)
beat_mel = librosa.util.sync(mel, beat_frames)
```

---

## Network Architecture

### BeatNet: Per-Frame Downbeat Classifier

```python
import torch
import torch.nn as nn

class BeatNet(nn.Module):
    """
    Fully convolutional network for per-frame downbeat detection.
    Based on KeyNet architecture but outputs per-frame probabilities.
    """

    def __init__(self, n_mels=80, n_classes=1):
        super().__init__()

        # Encoder (similar to KeyNet)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),  # Pool frequency only, preserve time
            nn.Dropout2d(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.25)
        )

        # After 3x frequency pooling: 80 / 8 = 10 frequency bins

        # Collapse frequency dimension, output per-frame logits
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(10, 1)),  # Collapse frequency
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, n_classes, kernel_size=(1, 1)),  # Per-frame output
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, n_mels, time_frames)
        Returns:
            logits: (batch, time_frames) — per-frame downbeat probability
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x.squeeze(1).squeeze(1)  # (batch, time_frames)
```

### Alternative: Temporal Convolution Network (TCN)

For better temporal modeling across longer time spans:

```python
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        return self.dropout(self.activation(self.norm(self.conv(x))))

class BeatTCN(nn.Module):
    """TCN for downbeat detection with large receptive field."""

    def __init__(self, n_mels=80):
        super().__init__()

        # Frequency reduction
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(1, 32, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((4, 1)),
            nn.Conv2d(32, 64, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((4, 1)),
            nn.Conv2d(64, 128, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, None))  # Collapse to (batch, 128, 1, time)
        )

        # Temporal convolutions with increasing dilation
        # Receptive field grows exponentially
        self.temporal = nn.Sequential(
            TemporalBlock(128, 128, kernel_size=3, dilation=1),
            TemporalBlock(128, 128, kernel_size=3, dilation=2),
            TemporalBlock(128, 128, kernel_size=3, dilation=4),
            TemporalBlock(128, 128, kernel_size=3, dilation=8),
            TemporalBlock(128, 128, kernel_size=3, dilation=16),
            TemporalBlock(128, 64, kernel_size=3, dilation=32),
        )

        self.output = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        x = self.freq_encoder(x)  # (batch, 128, 1, time)
        x = x.squeeze(2)  # (batch, 128, time)
        x = self.temporal(x)  # (batch, 64, time)
        x = self.output(x)  # (batch, 1, time)
        return x.squeeze(1)  # (batch, time)
```

---

## Dataset Preparation

### Required Data

For each track in the dataset:
1. **Audio file** (MP3, WAV, etc.)
2. **Accurate BPM** (already available)
3. **First beat position in seconds** (ground truth annotation)

### Label Generation

For per-frame classification, generate frame-level labels:

```python
def generate_labels(first_beat_sec, bpm, duration_sec, hop_length=512, sr=44100):
    """
    Generate per-frame downbeat labels.

    Args:
        first_beat_sec: Ground truth first beat position
        bpm: Track BPM
        duration_sec: Audio duration
        hop_length: Spectrogram hop length
        sr: Sample rate

    Returns:
        labels: Binary array, 1 at downbeat frames
    """
    beat_interval = 60.0 / bpm
    bar_interval = beat_interval * 4  # Assuming 4/4 time

    # Generate all downbeat times
    downbeat_times = []
    t = first_beat_sec
    while t < duration_sec:
        downbeat_times.append(t)
        t += bar_interval

    # Also go backwards for tracks that start mid-bar
    t = first_beat_sec - bar_interval
    while t >= 0:
        downbeat_times.append(t)
        t -= bar_interval

    downbeat_times = sorted(downbeat_times)

    # Convert to frames
    n_frames = int(duration_sec * sr / hop_length) + 1
    labels = np.zeros(n_frames, dtype=np.float32)

    # Mark downbeat frames (with small tolerance window)
    tolerance_frames = 3  # ~35ms at 512 hop
    for db_time in downbeat_times:
        frame = int(db_time * sr / hop_length)
        for f in range(frame - tolerance_frames, frame + tolerance_frames + 1):
            if 0 <= f < n_frames:
                labels[f] = 1.0

    return labels
```

### Data Augmentation

```python
def augment_audio(audio, sr, bpm, first_beat):
    """
    Augment training data while preserving beat alignment.
    """
    augmented = []

    # 1. Time stretch (adjust BPM proportionally)
    for rate in [0.95, 1.0, 1.05]:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        new_bpm = bpm * rate
        new_first_beat = first_beat / rate
        augmented.append((stretched, new_bpm, new_first_beat))

    # 2. Pitch shift (doesn't affect timing)
    for semitones in [-2, -1, 0, 1, 2]:
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        augmented.append((shifted, bpm, first_beat))

    # 3. Add noise
    noise = np.random.randn(len(audio)) * 0.005
    noisy = audio + noise
    augmented.append((noisy, bpm, first_beat))

    return augmented
```

### Dataset Class

```python
class FirstBeatDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, audio_dir, segment_duration=30.0):
        """
        Args:
            annotations_file: CSV with columns [filename, bpm, first_beat_sec]
            audio_dir: Directory containing audio files
            segment_duration: Length of audio segment to analyze
        """
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = Path(audio_dir)
        self.segment_duration = segment_duration
        self.sr = 44100
        self.hop_length = 512
        self.n_mels = 80

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # Load audio
        audio_path = self.audio_dir / row['filename']
        audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.segment_duration)

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels,
            hop_length=self.hop_length, fmin=20, fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

        # Generate labels
        labels = generate_labels(
            row['first_beat_sec'], row['bpm'],
            self.segment_duration, self.hop_length, self.sr
        )

        # Convert to tensors
        mel_tensor = torch.from_numpy(mel_db).unsqueeze(0).float()
        labels_tensor = torch.from_numpy(labels).float()

        return mel_tensor, labels_tensor
```

---

## Training

### Loss Function

For per-frame classification with class imbalance (most frames are not downbeats):

```python
# Option 1: Weighted BCE
pos_weight = torch.tensor([20.0])  # Downbeats are ~5% of frames
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Option 2: Focal Loss (better for extreme imbalance)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
```

### Training Loop

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for mel, labels in dataloader:
        mel = mel.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(mel)

        # Align lengths (spectrogram might have different length than labels)
        min_len = min(logits.shape[-1], labels.shape[-1])
        loss = criterion(logits[:, :min_len], labels[:, :min_len])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
```

### Hyperparameters

```python
config = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 100,
    'early_stopping_patience': 10,
    'segment_duration': 30.0,  # seconds
    'hop_length': 512,
    'n_mels': 80,
    'sr': 44100,
}
```

---

## Inference & Post-Processing

### Raw Inference

```python
def predict_downbeats(model, audio_path, device='cpu'):
    """Get per-frame downbeat probabilities."""
    model.eval()

    # Load and preprocess
    audio, sr = librosa.load(audio_path, sr=44100, duration=30.0)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=80, hop_length=512, fmin=20, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    mel_tensor = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    return probs
```

### Finding First Beat with Known BPM

```python
def find_first_beat(probs, bpm, sr=44100, hop_length=512):
    """
    Find first beat by scoring candidate positions.

    Args:
        probs: Per-frame downbeat probabilities
        bpm: Known BPM
        sr: Sample rate
        hop_length: Spectrogram hop length

    Returns:
        first_beat_sec: Estimated first beat position
    """
    frame_duration = hop_length / sr
    bar_interval = 60.0 / bpm * 4  # 4 beats per bar
    bar_frames = int(bar_interval / frame_duration)

    # Score each possible phase offset
    best_score = -1
    best_offset = 0

    # Try each frame in the first bar as potential downbeat
    for offset in range(min(bar_frames, len(probs))):
        # Sum probabilities at all downbeat positions for this phase
        score = 0
        frame = offset
        count = 0
        while frame < len(probs):
            score += probs[frame]
            count += 1
            frame += bar_frames

        score /= count  # Normalize by number of downbeats checked

        if score > best_score:
            best_score = score
            best_offset = offset

    first_beat_sec = best_offset * frame_duration
    return first_beat_sec, best_score
```

### Handling Intros/Silence

```python
def find_first_musical_beat(probs, bpm, sr=44100, hop_length=512, silence_threshold=0.1):
    """
    Find first beat, accounting for silent intros.
    """
    frame_duration = hop_length / sr

    # Find where audio "starts" (first frame with significant probability)
    music_start_frame = 0
    for i, p in enumerate(probs):
        if p > silence_threshold:
            music_start_frame = i
            break

    # Only consider downbeats after music starts
    bar_interval = 60.0 / bpm * 4
    bar_frames = int(bar_interval / frame_duration)

    # Score phases, but only count frames after music_start_frame
    best_score = -1
    best_offset = 0

    for offset in range(bar_frames):
        score = 0
        count = 0
        frame = offset
        while frame < len(probs):
            if frame >= music_start_frame:
                score += probs[frame]
                count += 1
            frame += bar_frames

        if count > 0:
            score /= count
            if score > best_score:
                best_score = score
                best_offset = offset

    first_beat_sec = best_offset * frame_duration
    return first_beat_sec, best_score
```

---

## Evaluation Metrics

### Beat Alignment Accuracy

Since beats repeat at BPM intervals, we measure alignment modulo beat interval:

```python
def evaluate_beat_accuracy(predicted_sec, ground_truth_sec, bpm, tolerance_ms=50):
    """
    Evaluate first beat prediction accuracy.

    A prediction is "correct" if it falls on the same beat grid as ground truth
    (accounting for the periodic nature of beats).
    """
    beat_interval = 60.0 / bpm
    bar_interval = beat_interval * 4

    # Compute offset from ground truth, modulo bar interval
    diff = abs(predicted_sec - ground_truth_sec)
    diff_mod_bar = diff % bar_interval

    # Check if within tolerance (or one bar - tolerance for wrap-around)
    tolerance_sec = tolerance_ms / 1000.0

    if diff_mod_bar < tolerance_sec or (bar_interval - diff_mod_bar) < tolerance_sec:
        return 'correct'

    # Check if off by 1, 2, or 3 beats
    for beat_offset in [1, 2, 3]:
        expected_diff = beat_interval * beat_offset
        if abs(diff_mod_bar - expected_diff) < tolerance_sec:
            return f'off_by_{beat_offset}_beats'

    return 'wrong'

def compute_metrics(predictions, ground_truths, bpms):
    """Compute accuracy metrics over dataset."""
    results = {'correct': 0, 'off_by_1_beats': 0, 'off_by_2_beats': 0,
               'off_by_3_beats': 0, 'wrong': 0}

    for pred, gt, bpm in zip(predictions, ground_truths, bpms):
        result = evaluate_beat_accuracy(pred, gt, bpm)
        results[result] += 1

    total = len(predictions)
    return {k: v / total * 100 for k, v in results.items()}
```

---

## Implementation Roadmap

### Phase 1: Data Preparation
1. [ ] Export dataset: filename, BPM, first_beat_sec to CSV
2. [ ] Verify ground truth annotations are accurate
3. [ ] Split into train/validation/test sets (80/10/10)
4. [ ] Implement `FirstBeatDataset` class

### Phase 2: Model Development
1. [ ] Implement `BeatNet` architecture
2. [ ] Train on dataset with focal loss
3. [ ] Evaluate on test set
4. [ ] Iterate on architecture if needed

### Phase 3: Integration
1. [ ] Export trained model to TorchScript or ONNX
2. [ ] Integrate with lexicon-beatdetector as alternative/ensemble
3. [ ] Benchmark against current DSP approach
4. [ ] A/B test on production data

### Phase 4: Optimization
1. [ ] Model pruning/quantization for faster inference
2. [ ] Optimize for browser (ONNX.js or TensorFlow.js)
3. [ ] Or keep as server-side component (like openkeyscan-analyzer)

---

## Expected Results

Based on similar research and the key detection results:

| Metric | DSP Approach (current) | CNN Approach (expected) |
|--------|----------------------|------------------------|
| Exact match | ~60-70% | ~75-85% |
| Within 1 beat | ~75-80% | ~90-95% |
| Within 4 beats | ~85-90% | ~95-98% |

The CNN should excel at:
- Tracks with complex intros
- Genres with syncopation
- Tracks where the "musical start" is ambiguous

The CNN may struggle with:
- Very long silent intros (>30 seconds)
- Non-4/4 time signatures (unless trained on them)
- Genres not represented in training data

---

## References

### Research Papers
- Böck, Krebs, Widmer. "Joint Beat and Downbeat Tracking with Recurrent Neural Networks" (ISMIR 2016)
- Fuentes, McFee, Crayencour, Cabral, Bello. "A Data-Driven Approach to Mid-level Perceptual Musical Feature Modeling" (ISMIR 2019)
- Davies, Plumbley. "Context-Dependent Beat Tracking of Musical Audio" (IEEE TASLP 2007)

### Existing Implementations
- [madmom](https://github.com/CPJKU/madmom) — Python beat tracking library with RNN models
- [BeatNet](https://github.com/mjhydri/BeatNet) — Real-time beat tracking
- [librosa](https://librosa.org/) — Beat tracking baseline

---

*Created: 2025-02-03*
