#!/usr/bin/env python3
"""
Performance comparison: CPU vs MPS (Apple Silicon GPU)
"""
import subprocess
import sys
import json
import time
from pathlib import Path

# Test file
TEST_FILE = Path("/Volumes/Storage/Music/Luca Agnelli - Apollo.mp3")

if not TEST_FILE.exists():
    print(f"Test file not found: {TEST_FILE}")
    sys.exit(1)

project_root = Path(__file__).parent.parent

def test_device(device_name, num_runs=3):
    """Test inference performance on specified device."""
    print(f"\n{'='*70}")
    print(f"Testing with device: {device_name.upper()}")
    print(f"{'='*70}")

    # Start server with specified device
    process = subprocess.Popen(
        ['pipenv', 'run', 'python', 'openkeyscan_analyzer_server.py', '--device', device_name, '-m', 'checkpoints/keynet.pt'],
        cwd=project_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Wait for ready signal
    ready = False
    while True:
        line = process.stderr.readline()
        if not line:
            break
        if 'Server ready' in line:
            ready = True
            break

    if not ready:
        print(f"❌ Server failed to start on {device_name}")
        process.terminate()
        return None

    print(f"✅ Server ready on {device_name}")

    # Run inference multiple times
    times = []
    for i in range(num_runs):
        request = {
            'id': f'test-{i}',
            'path': str(TEST_FILE)
        }

        start = time.time()
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()

        # Read response
        response_line = process.stdout.readline()
        elapsed = time.time() - start

        if response_line:
            response = json.loads(response_line)
            if response.get('status') == 'success':
                times.append(elapsed)
                result = f"{response['camelot']} ({response['key']})"
                print(f"  Run {i+1}: {elapsed:.3f}s - Result: {result}")
            else:
                print(f"  Run {i+1}: Error - {response.get('error', 'Unknown')}")
        else:
            print(f"  Run {i+1}: No response")

    # Cleanup
    process.terminate()
    process.wait(timeout=2)

    if times:
        avg_time = sum(times) / len(times)
        print(f"\n  Average: {avg_time:.3f}s ({len(times)} successful runs)")
        return avg_time
    else:
        return None

# Run tests
print("="*70)
print("MPS vs CPU Performance Comparison")
print("="*70)
print(f"Test file: {TEST_FILE.name}")

cpu_time = test_device('cpu', num_runs=3)
mps_time = test_device('mps', num_runs=3)

# Summary
print("\n" + "="*70)
print("PERFORMANCE SUMMARY")
print("="*70)

if cpu_time and mps_time:
    speedup = cpu_time / mps_time
    print(f"CPU:  {cpu_time:.3f}s")
    print(f"MPS:  {mps_time:.3f}s")
    print(f"\n🚀 Speedup: {speedup:.2f}x faster on MPS")

    if speedup > 1.5:
        print("✅ Significant performance improvement!")
    elif speedup > 1.0:
        print("✅ MPS is faster")
    else:
        print("⚠️  CPU is faster (model may be too small to benefit from GPU)")
else:
    print("❌ Could not complete performance comparison")

print("="*70)
