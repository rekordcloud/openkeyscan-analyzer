#!/usr/bin/env python3
"""
Key Detection Server - stdin/stdout JSON Protocol

Runs as a long-running process, loading the model once and keeping it in memory.
Communicates via line-delimited JSON (NDJSON) protocol.

Protocol:
  Request:  {"id": "uuid", "path": "/absolute/path/file.mp3"}
  Success:  {"id": "uuid", "status": "success", "camelot": "9A", "key": "E minor", "class_id": 8, "filename": "file.mp3"}
  Error:    {"id": "uuid", "status": "error", "error": "Error message", "filename": "file.mp3"}
  Ready:    {"type": "ready"}
  Heartbeat: {"type": "heartbeat"}
"""

import sys
import json
import torch
import librosa
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Import from existing modules
from dataset import CAMELOT_MAPPING
from eval import load_model

# Get resource path helper from predict_keys
def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        base_path = Path(__file__).parent
    return base_path / relative_path


def preprocess_mp3(mp3_path, sample_rate=44100, n_bins=105, hop_length=8820):
    """
    Loads an mp3, converts to mono, resamples, and extracts a log-magnitude CQT spectrogram.

    Args:
        mp3_path (Path): Path to .mp3 file.
        sample_rate (int): Target sampling rate.
        n_bins (int): Number of CQT bins.
        hop_length (int): Hop length for CQT.

    Returns:
        torch.Tensor: Shape (1, freq_bins, time_frames), ready for model input.
    """
    # Use librosa to load and resample audio
    waveform, sr = librosa.load(mp3_path, sr=sample_rate, mono=True)
    waveform = waveform.astype(np.float32)

    cqt = librosa.cqt(waveform, sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=24, fmin=65)
    spec = np.abs(cqt)
    spec = np.log1p(spec)

    # Remove last frequency bin
    chunk = spec[:, 0:-2]
    spec_tensor = torch.tensor(chunk, dtype=torch.float32)
    if spec_tensor.ndim == 2:
        spec_tensor = spec_tensor.unsqueeze(0)  # Shape: (1, freq, time)
    return spec_tensor


def camelot_output(pred_camelot):
    """
    Formats the Camelot prediction.

    Args:
        pred_camelot (int): 0-23, neural network output

    Returns:
        (str, str): camelot_str (e.g. "6A"), key_text (e.g. "D minor")
    """
    idx = (pred_camelot % 12) + 1
    mode = "A" if pred_camelot < 12 else "B"
    camelot_str = f"{idx}{mode}"

    names = [k for k, v in CAMELOT_MAPPING.items() if v == pred_camelot]
    if names:
        key_text = "/".join(sorted(set(names)))
    else:
        key_text = "Unknown"
    return camelot_str, key_text


def camelot_to_openkey(camelot_str):
    """
    Converts Camelot notation to Open Key notation.

    Camelot uses: 1A-12A (minor), 1B-12B (major)
    Open Key uses: 1m-12m (minor), 1d-12d (major)

    Both minor and major keys use the same offset because relative
    keys share the same position (e.g., 8A = A minor, 8B = C major).

    Args:
        camelot_str (str): Camelot notation (e.g. "9A", "3B")

    Returns:
        str: Open Key notation (e.g. "2m", "1d")

    Examples:
        8A (A minor) = 1m
        8B (C major) = 1d
        9A (E minor) = 2m
    """
    # Parse Camelot notation
    camelot_num = int(camelot_str[:-1])
    camelot_mode = camelot_str[-1]

    # Both minor and major use the same offset
    openkey_num = ((camelot_num - 8) % 12) + 1
    openkey_mode = "m" if camelot_mode == "A" else "d"

    return f"{openkey_num}{openkey_mode}"


class KeyDetectionServer:
    """
    Server that processes key detection requests via stdin/stdout.

    Each worker thread loads its own model instance to avoid thread safety issues.
    """

    def __init__(self, model_path=None, device=None, num_workers=1):
        """
        Initialize the server.

        Args:
            model_path (str): Path to model checkpoint
            device (str): Device to use ('cpu' or 'cuda')
            num_workers (int): Number of worker threads (default: 1, each gets own model)
        """
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # Store model path and device for per-thread loading
        if model_path is None:
            self.model_path = get_resource_path('checkpoints/keynet.pt')
        else:
            self.model_path = model_path

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Thread-local storage for per-thread model instances
        self.thread_local = threading.local()

        # Log configuration
        print(f"Server configuration:", file=sys.stderr)
        print(f"  Workers: {self.num_workers}", file=sys.stderr)
        print(f"  Model path: {self.model_path}", file=sys.stderr)
        print(f"  Device: {self.device}", file=sys.stderr)
        print(f"  Strategy: One model instance per worker thread", file=sys.stderr)

        self.running = True

    def get_model(self):
        """
        Get the model instance for the current thread.
        Loads the model on first access per thread (lazy loading).

        Returns:
            KeyNet: The model instance for this thread
        """
        if not hasattr(self.thread_local, 'model'):
            # This thread doesn't have a model yet - load it
            thread_id = threading.current_thread().name
            print(f"[Thread {thread_id}] Loading model from {self.model_path}...", file=sys.stderr)
            self.thread_local.model = load_model(self.model_path, self.device)
            print(f"[Thread {thread_id}] Model loaded on {self.device}", file=sys.stderr)

        return self.thread_local.model

    def send_message(self, message):
        """Send a JSON message to stdout."""
        try:
            json_str = json.dumps(message)
            print(json_str, flush=True)
        except Exception as e:
            print(f"Error sending message: {e}", file=sys.stderr)

    def process_request(self, request):
        """
        Process a single key detection request.

        Args:
            request (dict): Request with 'id' and 'path' keys

        Returns:
            dict: Response message
        """
        request_id = request.get('id', 'unknown')
        file_path = request.get('path', '')

        try:
            mp3_path = Path(file_path)

            if not mp3_path.exists():
                return {
                    'id': request_id,
                    'status': 'error',
                    'error': 'File not found',
                    'filename': mp3_path.name
                }

            if mp3_path.suffix.lower() != '.mp3':
                return {
                    'id': request_id,
                    'status': 'error',
                    'error': 'Not an MP3 file',
                    'filename': mp3_path.name
                }

            # Preprocess audio
            spec_tensor = preprocess_mp3(mp3_path)
            spec_tensor = spec_tensor.to(self.device)
            spec_tensor = spec_tensor.unsqueeze(0)  # Add batch dimension

            # Get thread-local model instance and run inference
            model = self.get_model()
            with torch.no_grad():
                outputs = model(spec_tensor)
                pred = int(torch.argmax(outputs, dim=1).cpu().item())

            # Format output
            camelot_str, key_text = camelot_output(pred)
            openkey_str = camelot_to_openkey(camelot_str)

            return {
                'id': request_id,
                'status': 'success',
                'camelot': camelot_str,
                'openkey': openkey_str,
                'key': key_text,
                'class_id': pred,
                'filename': mp3_path.name
            }

        except Exception as e:
            return {
                'id': request_id,
                'status': 'error',
                'error': str(e),
                'filename': Path(file_path).name if file_path else 'unknown'
            }

    def handle_request(self, line):
        """Parse and handle a request line."""
        try:
            request = json.loads(line)

            # Process the request
            response = self.process_request(request)
            self.send_message(response)

        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error handling request: {e}", file=sys.stderr)

    def run(self):
        """Main server loop - read from stdin and process requests."""
        # Send ready message
        self.send_message({'type': 'ready'})
        print("Server ready, waiting for requests...", file=sys.stderr)

        # Start heartbeat thread
        def heartbeat():
            while self.running:
                time.sleep(30)
                if self.running:
                    self.send_message({'type': 'heartbeat'})

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        # Process requests from stdin
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                # Submit to thread pool for concurrent processing
                self.executor.submit(self.handle_request, line)

        except KeyboardInterrupt:
            print("Shutting down...", file=sys.stderr)
        finally:
            self.running = False
            self.executor.shutdown(wait=True)
            print("Server stopped", file=sys.stderr)


def main():
    """Entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Key Detection Server (stdin/stdout JSON protocol)")
    parser.add_argument('-m', '--model_path', type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help="Number of worker threads (default: 1, each loads own model instance)")

    args = parser.parse_args()

    server = KeyDetectionServer(
        model_path=args.model_path,
        device=args.device,
        num_workers=args.workers
    )

    server.run()


if __name__ == '__main__':
    main()
