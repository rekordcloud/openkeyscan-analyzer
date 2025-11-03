#!/usr/bin/env python3
"""
Test harness for key detection server with memory tracking.

Spawns openkeyscan_analyzer_server.py and sends requests for random MP3 files.
Logs results in format: "Filename.mp3: Result key"

Usage:
    python test_server.py                  # Test with 10 files (default)
    python test_server.py -n 5             # Test with 5 files
    python test_server.py -n 20            # Test with 20 files
    python test_server.py -d ~/Music/test  # Use different directory
"""

import subprocess
import json
import sys
import uuid
import glob
import os
import random
import time
import threading
import argparse
from pathlib import Path

# Try to import psutil for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    import resource


class ServerTester:
    """Test harness for the key detection server."""

    def __init__(self, server_script='openkeyscan_analyzer_server.py', workers=1):
        """
        Initialize the tester.

        Args:
            server_script (str): Path to the server script
            workers (int): Number of worker threads for the server
        """
        self.server_script = server_script
        self.workers = workers
        self.process = None
        self.results = {}
        self.results_lock = threading.Lock()
        self.server_ready = threading.Event()
        self.peak_memory = 0

    def start_server(self):
        """Start the server process."""
        print(f"Starting key detection server with {self.workers} worker(s)...")
        sys.stdout.flush()

        # Use venv python if available, otherwise system python
        python_cmd = './venv/bin/python3' if os.path.exists('./venv/bin/python3') else 'python3'

        self.process = subprocess.Popen(
            [python_cmd, self.server_script, '-w', str(self.workers)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        # Start thread to read responses
        self.response_thread = threading.Thread(target=self._read_responses, daemon=True)
        self.response_thread.start()

        # Start thread to read stderr
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()

    def _read_responses(self):
        """Read and parse responses from server stdout."""
        try:
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)

                    if message.get('type') == 'ready':
                        print("Server ready!")
                        self.server_ready.set()
                    elif message.get('type') == 'heartbeat':
                        # Ignore heartbeats during testing
                        pass
                    elif 'id' in message:
                        # Store result
                        with self.results_lock:
                            self.results[message['id']] = message
                    else:
                        print(f"Unknown message: {message}")

                except json.JSONDecodeError as e:
                    print(f"Failed to parse response: {line}")
                    print(f"Error: {e}")

        except Exception as e:
            print(f"Error reading responses: {e}")

    def _read_stderr(self):
        """Read and display stderr from server."""
        try:
            for line in self.process.stderr:
                print(f"[SERVER] {line.rstrip()}")
        except Exception as e:
            print(f"Error reading stderr: {e}")

    def send_request(self, request_id, file_path):
        """
        Send a key detection request to the server.

        Args:
            request_id (str): Unique request ID
            file_path (str): Absolute path to MP3 file
        """
        request = {
            'id': request_id,
            'path': str(file_path)
        }

        try:
            json_str = json.dumps(request)
            self.process.stdin.write(json_str + '\n')
            self.process.stdin.flush()
        except Exception as e:
            print(f"Error sending request: {e}")

    def wait_for_result(self, request_id, timeout=30):
        """
        Wait for a specific result.

        Args:
            request_id (str): Request ID to wait for
            timeout (int): Timeout in seconds

        Returns:
            dict or None: Result message or None if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.results_lock:
                if request_id in self.results:
                    return self.results[request_id]
            time.sleep(0.1)
        return None

    def get_memory_usage(self):
        """
        Get current memory usage of the server process and this process.

        Returns:
            dict: Memory usage in MB
        """
        memory_info = {
            'server_rss_mb': 0,
            'server_vms_mb': 0,
            'client_rss_mb': 0,
            'total_rss_mb': 0
        }

        if PSUTIL_AVAILABLE:
            # Get server process memory
            if self.process:
                try:
                    server_proc = psutil.Process(self.process.pid)
                    server_mem = server_proc.memory_info()
                    memory_info['server_rss_mb'] = server_mem.rss / 1024 / 1024
                    memory_info['server_vms_mb'] = server_mem.vms / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Get client (this) process memory
            client_proc = psutil.Process()
            client_mem = client_proc.memory_info()
            memory_info['client_rss_mb'] = client_mem.rss / 1024 / 1024
            memory_info['total_rss_mb'] = memory_info['server_rss_mb'] + memory_info['client_rss_mb']

            # Track peak
            if memory_info['total_rss_mb'] > self.peak_memory:
                self.peak_memory = memory_info['total_rss_mb']
        else:
            # Fallback to resource module (less accurate)
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                memory_info['client_rss_mb'] = usage.ru_maxrss / 1024  # macOS reports in bytes
                memory_info['total_rss_mb'] = memory_info['client_rss_mb']
            except:
                pass

        return memory_info

    def print_memory_usage(self, label=""):
        """Print current memory usage."""
        mem = self.get_memory_usage()
        if PSUTIL_AVAILABLE:
            print(f"[MEMORY {label}] Server: {mem['server_rss_mb']:.1f}MB | "
                  f"Client: {mem['client_rss_mb']:.1f}MB | "
                  f"Total: {mem['total_rss_mb']:.1f}MB")
        else:
            print(f"[MEMORY {label}] Approx: {mem['total_rss_mb']:.1f}MB")

    def stop_server(self):
        """Stop the server process."""
        if self.process:
            self.process.stdin.close()
            self.process.wait(timeout=5)
            print("Server stopped")

    def find_random_mp3s(self, directory, count=10):
        """
        Find random MP3 files from a directory.

        Args:
            directory (str): Directory to search (supports ~)
            count (int): Number of files to return

        Returns:
            list: List of absolute paths to MP3 files
        """
        expanded_dir = os.path.expanduser(directory)

        if not os.path.exists(expanded_dir):
            print(f"Warning: Directory {expanded_dir} does not exist")
            return []

        # Find all MP3 files recursively
        pattern = os.path.join(expanded_dir, '**', '*.mp3')
        mp3_files = glob.glob(pattern, recursive=True)

        if not mp3_files:
            print(f"Warning: No MP3 files found in {expanded_dir}")
            return []

        # Randomly select files
        selected = random.sample(mp3_files, min(count, len(mp3_files)))
        return selected


def main():
    """Main test function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test key detection server with memory tracking")
    parser.add_argument('-n', '--num-files', type=int, default=10,
                        help="Number of random MP3 files to test (default: 10)")
    parser.add_argument('-d', '--directory', type=str, default='~/Music/spotify',
                        help="Directory to search for MP3 files (default: ~/Music/spotify)")
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help="Number of worker threads for the server (default: 1)")
    args = parser.parse_args()

    # Print memory tracking availability
    if not PSUTIL_AVAILABLE:
        print("Note: psutil not available - limited memory tracking")
        print("Install with: pip install psutil")
        print()

    tester = ServerTester(workers=args.workers)

    try:
        # Start server
        tester.start_server()

        # Wait for server to be ready
        if not tester.server_ready.wait(timeout=10):
            print("Error: Server did not start within 10 seconds")
            return

        # Check memory after server startup
        tester.print_memory_usage("after startup")
        print()

        # Find random MP3 files
        mp3_directory = args.directory
        num_files = args.num_files

        print(f"Finding {num_files} random MP3 files from {mp3_directory}...")
        mp3_files = tester.find_random_mp3s(mp3_directory, num_files)

        if not mp3_files:
            print("No MP3 files found. Exiting.")
            return

        print(f"Found {len(mp3_files)} files to analyze")
        print()

        # Send requests
        print(f"Sending {len(mp3_files)} requests...\n")
        requests = {}
        start_time = time.time()

        for mp3_file in mp3_files:
            request_id = str(uuid.uuid4())
            requests[request_id] = mp3_file
            tester.send_request(request_id, mp3_file)

        # Check memory after sending all requests
        tester.print_memory_usage("after sending requests")

        # Wait for all results
        print("Waiting for results...\n")
        print("Results:")
        print("-" * 70)

        successful = 0
        failed = 0
        file_count = 0

        for request_id, file_path in requests.items():
            result = tester.wait_for_result(request_id, timeout=60)
            file_count += 1

            if result:
                filename = Path(file_path).name
                if result['status'] == 'success':
                    camelot = result['camelot']
                    openkey = result.get('openkey', '')
                    key = result['key']
                    openkey_str = f" / {openkey}" if openkey else ""
                    print(f"{filename}: {camelot}{openkey_str} ({key})")
                    successful += 1
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"{filename}: ERROR - {error}")
                    failed += 1
            else:
                filename = Path(file_path).name
                print(f"{filename}: TIMEOUT - No response received")
                failed += 1

            # Track memory after each file
            mem = tester.get_memory_usage()
            if PSUTIL_AVAILABLE:
                print(f"    [Memory after file {file_count}/{len(requests)}] Server: {mem['server_rss_mb']:.1f}MB | Total: {mem['total_rss_mb']:.1f}MB")

        elapsed = time.time() - start_time

        # Check memory after processing
        tester.print_memory_usage("after processing")

        # Summary
        print("-" * 70)
        print(f"\nProcessed {successful + failed} files in {elapsed:.2f}s")
        print(f"Success: {successful}, Failed: {failed}")
        if successful > 0:
            print(f"Average: {elapsed / (successful + failed):.2f}s per file")

        # Memory summary
        if PSUTIL_AVAILABLE:
            print(f"Peak memory usage: {tester.peak_memory:.1f}MB")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("\nShutting down server...")
        tester.stop_server()
        print("Test complete")


if __name__ == '__main__':
    main()
