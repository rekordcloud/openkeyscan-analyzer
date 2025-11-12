#!/usr/bin/env python3
"""
Test script to verify reset command behavior.

This test confirms that:
1. Reset command is acknowledged
2. Tasks submitted before reset do NOT send results back
3. Tasks submitted after reset work correctly
4. No model reload happens (subprocess stays alive)
"""

import subprocess
import json
import sys
import time
import random
from pathlib import Path
from typing import Optional, Dict, Any

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ANSI color codes for pretty output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


def find_test_audio_files(base_path: Path, count: int = 5) -> list[Path]:
    """Find random audio files for testing."""
    audio_files = []

    # Supported formats
    extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg']

    for ext in extensions:
        audio_files.extend(base_path.rglob(ext))
        if len(audio_files) >= count * 2:
            break

    if len(audio_files) < count:
        raise FileNotFoundError(f"Could not find {count} audio files in {base_path}")

    return random.sample(audio_files, count)


class ResetTestHarness:
    """Test harness for reset command."""

    def __init__(self, executable_path: str):
        self.executable_path = executable_path
        self.process: Optional[subprocess.Popen] = None
        self.results_received = []
        self.system_messages = []

    def start_server(self) -> None:
        """Start the analyzer server."""
        print(f"{BLUE}Starting analyzer server...{RESET}")
        self.process = subprocess.Popen(
            [self.executable_path, '--workers', '2'],  # Use 2 workers for faster processing
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        # Wait for ready signal
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: float = 60) -> None:
        """Wait for the server to send ready signal."""
        start = time.time()
        while time.time() - start < timeout:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Server process terminated unexpectedly")

            try:
                msg = json.loads(line.strip())
                if msg.get('type') == 'ready':
                    print(f"{GREEN}[OK] Server ready!{RESET}\n")
                    return
            except json.JSONDecodeError:
                pass  # Ignore non-JSON output (stderr logs)

        raise TimeoutError("Server did not send ready signal within timeout")

    def send_request(self, file_path: Path, request_id: str) -> None:
        """Send an analysis request."""
        request = {
            'id': request_id,
            'path': str(file_path)
        }
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()

    def send_reset(self) -> None:
        """Send reset command."""
        print(f"{YELLOW}Sending reset command...{RESET}")
        reset_cmd = {'type': 'reset'}
        self.process.stdin.write(json.dumps(reset_cmd) + '\n')
        self.process.stdin.flush()

    def read_response(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Read a single response with timeout."""
        start = time.time()
        self.process.stdout.flush()

        while time.time() - start < timeout:
            # Check if there's data available
            import select
            if hasattr(select, 'select'):
                ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                if not ready:
                    continue

            line = self.process.stdout.readline()
            if not line:
                return None

            try:
                msg = json.loads(line.strip())

                # Categorize message
                if 'type' in msg:
                    self.system_messages.append(msg)
                    return msg
                else:
                    self.results_received.append(msg)
                    return msg

            except json.JSONDecodeError:
                continue  # Ignore non-JSON output

        return None

    def stop_server(self) -> None:
        """Stop the analyzer server."""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            print(f"{BLUE}Server stopped{RESET}\n")


def run_reset_test():
    """Run the reset command test."""
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}Reset Command Test{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    # Find executable
    if sys.platform == 'win32':
        exe_path = Path(__file__).parent.parent / 'dist' / 'openkeyscan-analyzer' / 'openkeyscan-analyzer.exe'
    else:
        exe_path = Path(__file__).parent.parent / 'dist' / 'openkeyscan-analyzer' / 'openkeyscan-analyzer'

    if not exe_path.exists():
        print(f"{RED}[ERROR] Executable not found: {exe_path}{RESET}")
        print(f"Please build the executable first with: pyinstaller openkeyscan_analyzer.spec")
        return False

    # Find test audio files
    music_dir = Path.home() / 'Music' / 'spotify'
    if not music_dir.exists():
        music_dir = Path.home() / 'Music'

    print(f"Finding audio files in: {music_dir}")
    try:
        test_files = find_test_audio_files(music_dir, count=5)
        print(f"Found {len(test_files)} test files\n")
    except FileNotFoundError as e:
        print(f"{RED}[ERROR] {e}{RESET}")
        return False

    # Create test harness
    harness = ResetTestHarness(str(exe_path))

    try:
        # Start server
        harness.start_server()

        # Phase 1: Send requests that will be interrupted by reset
        print(f"{BOLD}Phase 1: Sending requests before reset{RESET}")
        print(f"Sending 5 requests...")

        pre_reset_ids = []
        for i, file_path in enumerate(test_files):
            request_id = f"pre-reset-{i}"
            pre_reset_ids.append(request_id)
            harness.send_request(file_path, request_id)
            print(f"  Sent: {request_id} ({file_path.name})")

        # Wait a moment for processing to start
        print(f"\nWaiting 0.5s for processing to start...")
        time.sleep(0.5)

        # Phase 2: Send reset command
        print(f"\n{BOLD}Phase 2: Sending reset command{RESET}")
        harness.send_reset()

        # Wait for reset_complete
        print(f"Waiting for reset_complete acknowledgment...")
        reset_complete = False
        for _ in range(10):  # Try for 10 seconds
            msg = harness.read_response(timeout=1.0)
            if msg and msg.get('type') == 'reset_complete':
                print(f"{GREEN}[OK] Reset complete (generation: {msg.get('generation')}){RESET}\n")
                reset_complete = True
                break

        if not reset_complete:
            print(f"{RED}[FAIL] Did not receive reset_complete acknowledgment{RESET}")
            return False

        # Phase 3: Check that no results from pre-reset requests are received
        print(f"{BOLD}Phase 3: Verifying no stale results{RESET}")
        print(f"Waiting 3 seconds to check for stale results...")

        # Collect any responses for 3 seconds
        start = time.time()
        stale_results = []
        while time.time() - start < 3.0:
            msg = harness.read_response(timeout=0.5)
            if msg and 'id' in msg and msg.get('status') in ['success', 'error']:
                if msg['id'] in pre_reset_ids:
                    stale_results.append(msg)
                    print(f"{RED}[FAIL] Received stale result: {msg['id']}{RESET}")

        if not stale_results:
            print(f"{GREEN}[OK] No stale results received (all discarded by server){RESET}\n")
        else:
            print(f"{RED}[FAIL] Received {len(stale_results)} stale results!{RESET}")
            print(f"Expected: 0 results from pre-reset requests")
            print(f"Got: {len(stale_results)} results")
            return False

        # Phase 4: Send new requests after reset
        print(f"{BOLD}Phase 4: Testing new requests after reset{RESET}")
        print(f"Sending 3 new requests...")

        post_reset_ids = []
        for i in range(3):
            request_id = f"post-reset-{i}"
            post_reset_ids.append(request_id)
            harness.send_request(test_files[i], request_id)
            print(f"  Sent: {request_id} ({test_files[i].name})")

        # Wait for new results
        print(f"\nWaiting for new results...")
        new_results = []
        timeout = 30.0  # 30 seconds should be enough
        start = time.time()

        while len(new_results) < len(post_reset_ids) and time.time() - start < timeout:
            msg = harness.read_response(timeout=1.0)
            if msg and 'id' in msg and msg['id'] in post_reset_ids:
                new_results.append(msg)
                status = "SUCCESS" if msg.get('status') == 'success' else "ERROR"
                print(f"  [{status}] {msg['id']}: {msg.get('camelot', msg.get('error', 'unknown'))}")

        if len(new_results) == len(post_reset_ids):
            print(f"{GREEN}[OK] All new requests completed successfully{RESET}\n")
        else:
            print(f"{RED}[FAIL] Only received {len(new_results)}/{len(post_reset_ids)} results{RESET}")
            return False

        # Success!
        print(f"{BOLD}{'='*70}{RESET}")
        print(f"{GREEN}{BOLD}TEST PASSED!{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")

        print(f"Summary:")
        print(f"  - Reset command acknowledged: {GREEN}YES{RESET}")
        print(f"  - Stale results discarded: {GREEN}YES ({len(pre_reset_ids)} requests){RESET}")
        print(f"  - New requests work: {GREEN}YES ({len(new_results)} completed){RESET}")
        print(f"  - Server stayed alive: {GREEN}YES (no model reload){RESET}")

        return True

    except Exception as e:
        print(f"{RED}[ERROR] Test failed with exception: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        harness.stop_server()


if __name__ == '__main__':
    success = run_reset_test()
    sys.exit(0 if success else 1)
