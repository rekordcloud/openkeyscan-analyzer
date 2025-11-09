# Musical Key CNN

This repository provides a full pipeline for musical key classification based on convolutional neural networks, inspired by the paper [\[1\]](#literature). It contains scripts for preprocessing, training, evaluation, and prediction of key labels on new music tracks, using the GiantSteps and GiantSteps-MTG datasets and the [Camelot Wheel](https://mixedinkey.com/camelot-wheel/) for key mapping.

---

## Table of Contents

- [Description](#description)
- [Setup and Installation](#setup-and-installation)
- [Key Prediction for Your Own Songs](#key-prediction-for-your-own-songs)
- [Dataset Preparation](#dataset-preparation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Literature](#literature)

---

## Description

This repository implements a CNN model for musical key detection. It provides scripts to:
- Preprocess datasets (extract CQT spectrograms, prepare annotations, augment with pitch shifts)
- Train the model from scratch
- Evaluate model performance with [MIREX key evaluation metrics](https://www.music-ir.org/mirex/wiki/2025:Audio_Key_Detection)
- Predict keys for audio files (supports MP3, MP4, WAV, FLAC, OGG, M4A, AAC, AIFF, AU)

---

## Setup and Installation

### Python Version Requirement

**This project requires Python 3.12.x** (not 3.13 or newer).

**Why Python 3.12?**

PyTorch (the deep learning framework this project uses) has different support for macOS architectures:
- **PyTorch 2.3.0+**: Only supports ARM64 (Apple Silicon), dropped Intel Mac (x86_64) support
- **PyTorch 2.2.2**: Last version with x86_64 wheels for Intel Macs
- **Python 3.13**: Only compatible with PyTorch 2.5.0+, which lacks x86_64 support

To build executables for **both ARM64 and x86_64 architectures**, we use:
- Python 3.12.x (supports PyTorch 2.2.2)
- PyTorch 2.2.2 (last version with both ARM64 and x86_64 wheels)

**Installing Python 3.12:**

Download the official macOS universal2 installer from [python.org/downloads/macos/](https://www.python.org/downloads/macos/):
- Look for Python 3.12.10 or later in the 3.12 series
- Download the "macOS 64-bit universal2 installer" (.pkg file)
- Install it (provides both ARM64 and x86_64 support)

After installation, verify:
```sh
python3.12 --version  # Should show Python 3.12.x
file $(which python3.12)  # Should show "universal binary with 2 architectures"
```

### Installation Steps

We recommend using Pipenv for dependency management and virtual environment handling:

**Install Pipenv** (if not already installed):
```sh
pip install pipenv
```

**Install all dependencies:**
```sh
pipenv install
```

This will create a virtual environment and install all required packages including PyTorch, torchaudio, librosa, numpy, and tqdm.

**Activate the virtual environment:**
```sh
pipenv shell
```

**Note:** PyTorch will be installed automatically. For CUDA-enabled GPUs, you may want to manually install the CUDA-specific version by following the instructions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

**Alternative (legacy):** If you prefer not to use Pipenv, you can still use the traditional venv + requirements.txt approach:
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Key Prediction for Your Own Songs

You can analyze audio files in multiple formats (MP3, MP4, WAV, FLAC, OGG, M4A, AAC, AIFF, AU) or entire folders using the provided model or your own trained model:

```sh
python openkeyscan_analyzer.py -f path/to/your_song.mp3
python openkeyscan_analyzer.py -f path/to/your/music_folder/
```

The script prints a summary table with:
- Filename
- Classification index (0-23)
- Index according to the Camelot Wheel (e.g., "8A" or "3B")
- The corresponding key

You can set the model checkpoint path with `-m path/to/your_model.pt` and the computation device with `--device cuda` or `--device cpu`.

## Building a Standalone Executable

You can create a standalone executable that bundles all dependencies and the trained model, making it easy to distribute and run on any macOS system without requiring Python installation.

**Prerequisites:**
- Pipenv environment set up (see [Setup and Installation](#setup-and-installation))
- All dependencies installed

**Steps:**

1. Install development dependencies (including PyInstaller):
   ```sh
   pipenv install --dev
   ```

2. Build the executable using the provided spec file:
   ```sh
   pyinstaller openkeyscan_analyzer.spec
   ```
   The build process automatically dereferences all symlinks, ensuring the distribution is portable and can be safely zipped or copied.

3. The executable will be created in the `dist/openkeyscan-analyzer/` folder. You can run the server directly:
   ```sh
   ./dist/openkeyscan-analyzer/openkeyscan-analyzer
   ```

4. The entire `dist/openkeyscan-analyzer/` folder can be copied to any macOS system and run without Python or any dependencies installed.

**Note:**
- The bundled application includes the trained model (`keynet.pt`), so you don't need to specify the model path
- The executable folder is approximately 780MB due to PyTorch, librosa, and scientific computing dependencies (numpy, scipy, scikit-learn)
- The spec file automatically replaces all symlinks with actual files during the build process, making the distribution fully portable
- On **macOS**, all supported audio formats (MP3, MP4, WAV, FLAC, OGG, M4A, AAC, AIFF, AU) use native Core Audio frameworks, so no additional dependencies are needed
- For **Linux/Windows** distributions, you would need to bundle FFmpeg binaries or ensure FFmpeg is installed on the target system for compressed format support

## Dataset Preparation

For training and evaluation, you need the following datasets:

- [GiantSteps MTG Key Dataset](https://github.com/GiantSteps/giantsteps-mtg-key-dataset) (Training)
- [GiantSteps Key Dataset](https://github.com/GiantSteps/giantsteps-key-dataset) (Evaluation)

Directory structure:
Place or symlink the datasets under the `Dataset/` folder:

```sh
Dataset/
    giantsteps-key-dataset/
    giantsteps-mtg-key-dataset/
```

## Preprocessing

Before training or evaluation, preprocess the datasets to generate CQT spectrograms for all tracks and pitch-shifted variants.

`python preprocess_data.py`

All resulting .pkl spectrograms are stored in subfolders of `Dataset/`.

## Training

To train a new key classification model on the MTG dataset, run:

`python train.py`

You can modify hyperparameters or training parameters by editing `train.py`.

## Evaluation

To evaluate a trained model (e.g., calculate MIREX scores on GiantSteps):

`python eval.py`

The output includes overall accuracy and weighted [MIREX scores](https://www.music-ir.org/mirex/wiki/2025:Audio_Key_Detection).

The following table contains the percentage ratios and the weighted Mirex scores:

| Method | Weighted | Correct | Fifth | Relative | Parallel | Other |
| ------ | -------- | ------- | ----- |--------- | -------- | ----- |
| `keynet.pt`| 73.51 | 66.72 | 8.11 | 6.79 | 3.48 | 14.90 |
| Mixed In Key 8.3 | 75.70 | 69.37 | 8.11 | 5.13 | 3.64 | 13.74 |
| RekordBox 7.12 | 65.53 | 56.79 | 11.92 | 5.96 | 4.97 | 20.36 |

## Literature

Please cite and refer to the original publication for scientific use and further reading:

- \[1\] F. Korzeniowski and G. Widmer. "Genre-Agnostic Key Classification With Convolutional Neural Networks".
In: *Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR)* (2018) [arXiv](https://arxiv.org/abs/1808.05340)

- \[2\] F. Korzeniowski and G. Widmer. "End-to-End Musical Key Estimation Using a Convolutional Neural Network". In: *Proceedings of the 25th European Signal Processing Conference* (2017) [arXiv](https://arxiv.org/abs/1706.02921)

---

## Credits

This project is based on and extends the original [MusicalKeyCNN](https://github.com/a1ex90/MusicalKeyCNN) implementation by [a1ex90](https://github.com/a1ex90).
