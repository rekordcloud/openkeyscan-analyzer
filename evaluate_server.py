#!/usr/bin/env python3
"""
Evaluate predict_keys_server.py on GiantSteps Key Dataset.

Runs the server, sends audio files, collects predictions, and compares
against ground truth using MIREX evaluation metrics.
"""

import sys
import json
import subprocess
import time
from pathlib import Path
from collections import defaultdict
import uuid

# Import existing evaluation functions
from dataset import CAMELOT_MAPPING
from eval import mirex_category

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available, memory tracking disabled", file=sys.stderr)


def load_ground_truth(dataset_dir, file_ids):
    """
    Load ground truth annotations from .key files.

    Args:
        dataset_dir (Path): Path to dataset directory
        file_ids (list): List of file IDs (e.g., ["1004923.LOFI", ...])

    Returns:
        dict: Mapping of file_id -> camelot_index
    """
    annotations_dir = dataset_dir / 'annotations' / 'key'
    ground_truth = {}

    for file_id in file_ids:
        key_file = annotations_dir / f"{file_id}.key"
        if not key_file.exists():
            print(f"Warning: No annotation for {file_id}", file=sys.stderr)
            continue

        key_str = key_file.read_text().strip()

        if key_str not in CAMELOT_MAPPING:
            print(f"Warning: Unknown key '{key_str}' for {file_id}", file=sys.stderr)
            continue

        ground_truth[file_id] = {
            'key_str': key_str,
            'camelot_idx': CAMELOT_MAPPING[key_str]
        }

    return ground_truth


def start_server(model_path=None, workers=1):
    """
    Start predict_keys_server.py as a subprocess.

    Args:
        model_path (str): Path to model checkpoint (optional)
        workers (int): Number of worker threads (default: 1)

    Returns:
        subprocess.Popen: Server process
    """
    cmd = [sys.executable, 'predict_keys_server.py', '-w', str(workers)]

    if model_path:
        cmd.extend(['-m', model_path])

    print(f"Starting server: {' '.join(cmd)}", file=sys.stderr)

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )

    return process


def wait_for_ready(process, timeout=30):
    """
    Wait for server to send 'ready' message.

    Args:
        process (subprocess.Popen): Server process
        timeout (int): Timeout in seconds

    Returns:
        bool: True if ready, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if not line:
            # Check if process died
            if process.poll() is not None:
                stderr = process.stderr.read()
                print(f"Server died during startup:\n{stderr}", file=sys.stderr)
                return False
            continue

        try:
            message = json.loads(line.strip())
            if message.get('type') == 'ready':
                print("Server ready!", file=sys.stderr)
                return True
        except json.JSONDecodeError:
            pass

    print(f"Timeout waiting for server (waited {timeout}s)", file=sys.stderr)
    return False


def send_request(process, request_id, file_path):
    """Send a prediction request to the server."""
    request = {
        'id': request_id,
        'path': str(file_path)
    }
    process.stdin.write(json.dumps(request) + '\n')
    process.stdin.flush()


def collect_responses(process, num_expected, timeout=300):
    """
    Collect prediction responses from the server.

    Args:
        process (subprocess.Popen): Server process
        num_expected (int): Number of responses to collect
        timeout (int): Timeout in seconds

    Returns:
        dict: Mapping of request_id -> response
    """
    responses = {}
    start_time = time.time()

    while len(responses) < num_expected:
        if time.time() - start_time > timeout:
            print(f"Timeout collecting responses (got {len(responses)}/{num_expected})", file=sys.stderr)
            break

        line = process.stdout.readline()
        if not line:
            # Check if process died
            if process.poll() is not None:
                stderr = process.stderr.read()
                print(f"Server died:\n{stderr}", file=sys.stderr)
                break
            continue

        try:
            message = json.loads(line.strip())

            # Skip system messages
            if message.get('type') in ['ready', 'heartbeat']:
                continue

            # Store response
            if 'id' in message:
                responses[message['id']] = message

        except json.JSONDecodeError as e:
            print(f"Invalid JSON from server: {e}", file=sys.stderr)

    return responses


def calculate_mirex_metrics(predictions, ground_truth):
    """
    Calculate MIREX evaluation metrics.

    Args:
        predictions (dict): request_id -> response
        ground_truth (dict): file_id -> ground_truth_info

    Returns:
        dict: MIREX scores and detailed results
    """
    counts = {"correct": 0, "fifth": 0, "relative": 0, "parallel": 0, "rest": 0}
    detailed_results = []

    # Map request IDs back to file IDs
    for request_id, response in predictions.items():
        if response['status'] != 'success':
            print(f"Skipping failed prediction: {response.get('error', 'Unknown error')}", file=sys.stderr)
            continue

        filename = response['filename']
        file_id = filename.replace('.mp3', '')

        if file_id not in ground_truth:
            print(f"No ground truth for {file_id}", file=sys.stderr)
            continue

        gt_info = ground_truth[file_id]
        pred_idx = response['class_id']
        gt_idx = gt_info['camelot_idx']

        # Calculate category
        category = mirex_category(pred_idx, gt_idx)
        counts[category] += 1

        # Store detailed result
        detailed_results.append({
            'file_id': file_id,
            'filename': filename,
            'predicted_key': response['key'],
            'predicted_camelot': response['camelot'],
            'predicted_openkey': response['openkey'],
            'ground_truth_key': gt_info['key_str'],
            'predicted_idx': pred_idx,
            'ground_truth_idx': gt_idx,
            'category': category,
            'correct': category == 'correct'
        })

    # Calculate normalized scores
    total = sum(counts.values())
    if total == 0:
        return None

    scores = {k: v/total for k, v in counts.items()}
    weighted = (
        scores["correct"] +
        0.5 * scores["fifth"] +
        0.3 * scores["relative"] +
        0.2 * scores["parallel"]
    )

    return {
        'counts': counts,
        'scores': scores,
        'weighted': weighted,
        'total': total,
        'detailed_results': detailed_results
    }


def print_evaluation_report(metrics, processing_time, memory_stats=None):
    """
    Print comprehensive evaluation report.

    Args:
        metrics (dict): MIREX metrics
        processing_time (float): Total processing time in seconds
        memory_stats (dict): Memory usage statistics (optional)
    """
    if metrics is None:
        print("\nNo valid predictions to evaluate!")
        return

    print("\n" + "="*70)
    print("{:^70}".format("GiantSteps Key Dataset Evaluation Results"))
    print("="*70)

    # MIREX scores
    print(f"\n{'Category':<15} | {'Count':>6} | {'Percentage':>10} | {'Weight':>6}")
    print("-"*70)
    print(f"{'Correct':<15} | {metrics['counts']['correct']:>6} | {metrics['scores']['correct']*100:>9.2f}% | 1.0")
    print(f"{'Fifth':<15} | {metrics['counts']['fifth']:>6} | {metrics['scores']['fifth']*100:>9.2f}% | 0.5")
    print(f"{'Relative':<15} | {metrics['counts']['relative']:>6} | {metrics['scores']['relative']*100:>9.2f}% | 0.3")
    print(f"{'Parallel':<15} | {metrics['counts']['parallel']:>6} | {metrics['scores']['parallel']*100:>9.2f}% | 0.2")
    print(f"{'Other':<15} | {metrics['counts']['rest']:>6} | {metrics['scores']['rest']*100:>9.2f}% | 0.0")
    print("-"*70)
    print(f"{'WEIGHTED SCORE':<15} |        | {metrics['weighted']*100:>9.2f}% |")
    print("="*70)

    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Samples evaluated: {metrics['total']}")
    print(f"  Total time: {processing_time:.2f}s")
    print(f"  Average per file: {processing_time/metrics['total']:.2f}s")
    print(f"  Throughput: {metrics['total']/processing_time*60:.1f} files/min")

    if memory_stats and PSUTIL_AVAILABLE:
        print(f"\nMemory Usage:")
        print(f"  Peak RSS: {memory_stats['peak_rss_mb']:.1f} MB")
        print(f"  Average RSS: {memory_stats['avg_rss_mb']:.1f} MB")

    # Error analysis
    errors = [r for r in metrics['detailed_results'] if not r['correct']]
    if errors:
        print(f"\n{'='*70}")
        print(f"Misclassifications ({len(errors)} total):")
        print(f"{'='*70}")

        # Group by category
        by_category = defaultdict(list)
        for error in errors:
            by_category[error['category']].append(error)

        for category in ['fifth', 'relative', 'parallel', 'rest']:
            if category in by_category:
                print(f"\n{category.upper()} errors ({len(by_category[category])}):")
                print(f"{'-'*70}")
                for error in by_category[category][:10]:  # Show first 10
                    print(f"  {error['file_id']}")
                    print(f"    Ground Truth: {error['ground_truth_key']}")
                    print(f"    Predicted:    {error['predicted_key']} ({error['predicted_camelot']}, {error['predicted_openkey']})")
                if len(by_category[category]) > 10:
                    print(f"  ... and {len(by_category[category]) - 10} more")

    print("\n" + "="*70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate key detection server on GiantSteps dataset")
    parser.add_argument('-n', '--num_files', type=int, default=50,
                        help="Number of files to evaluate (default: 50)")
    parser.add_argument('-m', '--model_path', type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help="Number of worker threads (default: 1)")

    args = parser.parse_args()

    # Paths
    dataset_dir = Path('datasets/giantsteps-key-dataset')
    audio_dir = dataset_dir / 'audio'

    # Get list of audio files
    audio_files = sorted(audio_dir.glob('*.mp3'))[:args.num_files]
    if not audio_files:
        print("Error: No audio files found!", file=sys.stderr)
        return 1

    print(f"\nEvaluating {len(audio_files)} files from GiantSteps Key Dataset")
    print(f"Workers: {args.workers}")

    # Load ground truth
    file_ids = [f.stem.replace('.mp3', '') for f in audio_files]
    ground_truth = load_ground_truth(dataset_dir, file_ids)
    print(f"Loaded ground truth for {len(ground_truth)} files\n")

    # Start server
    server = start_server(model_path=args.model_path, workers=args.workers)

    try:
        # Wait for server to be ready
        if not wait_for_ready(server, timeout=30):
            print("Failed to start server", file=sys.stderr)
            return 1

        # Track memory if psutil is available
        memory_samples = []
        if PSUTIL_AVAILABLE:
            server_process = psutil.Process(server.pid)

        # Send all requests
        print(f"Sending {len(audio_files)} requests...")
        request_map = {}  # request_id -> file_id
        start_time = time.time()

        for audio_file in audio_files:
            request_id = str(uuid.uuid4())
            request_map[request_id] = audio_file.stem.replace('.mp3', '')
            send_request(server, request_id, audio_file.absolute())

            # Sample memory
            if PSUTIL_AVAILABLE:
                mem_info = server_process.memory_info()
                memory_samples.append(mem_info.rss / 1024 / 1024)  # MB

        print("Waiting for responses...")

        # Collect responses
        responses = collect_responses(server, len(audio_files), timeout=300)
        processing_time = time.time() - start_time

        print(f"\nReceived {len(responses)}/{len(audio_files)} responses")

        # Calculate metrics
        metrics = calculate_mirex_metrics(responses, ground_truth)

        # Calculate memory stats
        memory_stats = None
        if PSUTIL_AVAILABLE and memory_samples:
            memory_stats = {
                'peak_rss_mb': max(memory_samples),
                'avg_rss_mb': sum(memory_samples) / len(memory_samples)
            }

        # Print report
        print_evaluation_report(metrics, processing_time, memory_stats)

        return 0

    finally:
        # Shutdown server
        server.terminate()
        server.wait(timeout=5)
        if server.poll() is None:
            server.kill()


if __name__ == '__main__':
    sys.exit(main())
