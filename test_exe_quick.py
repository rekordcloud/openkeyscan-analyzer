#!/usr/bin/env python3
"""Quick test for the executable with specific MP3 files."""

import subprocess
import json
import sys
import os
import uuid
import time
import threading

exe_path = os.path.abspath("dist/openkeyscan-analyzer/openkeyscan-analyzer.exe")

# Test files
test_files = [
    r"C:\Users\Chris\Music\Athys & Duster - Barfight.mp3",
    r"C:\Users\Chris\Music\Audio - Combust.mp3",
    r"C:\Users\Chris\Music\Balthazar & JackRock - Andromeda.mp3"
]

print(f"Starting executable: {exe_path}")
process = subprocess.Popen(
    [exe_path, '-w', '1'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

results = {}
ready = False

def read_stdout():
    global ready
    try:
        for line in process.stdout:
            line = line.strip()
            #print(f"[DEBUG stdout] {line[:200]}")
            #sys.stdout.flush()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get('type') == 'ready':
                    print("[OK] Server ready!")
                    sys.stdout.flush()
                    ready = True
                elif 'id' in msg:
                    msg['_recv_time'] = time.time()  # Track when we received response
                    #print(f"[DEBUG] Got result for id {msg['id'][:8]}...")
                    #sys.stdout.flush()
                    results[msg['id']] = msg
            except json.JSONDecodeError:
                print(f"Non-JSON stdout: {line}")
    except Exception as e:
        print(f"Error reading stdout: {e}")

def read_stderr():
    try:
        for line in process.stderr:
            print(f"[stderr] {line.rstrip()}")
    except Exception as e:
        print(f"Error reading stderr: {e}")

# Start reader threads
stdout_thread = threading.Thread(target=read_stdout, daemon=True)
stderr_thread = threading.Thread(target=read_stderr, daemon=True)
stdout_thread.start()
stderr_thread.start()

# Wait for ready
print("Waiting for server to be ready...")
timeout = time.time() + 10
while not ready and time.time() < timeout:
    time.sleep(0.1)

if not ready:
    print("ERROR: Server did not become ready in 10 seconds")
    process.kill()
    sys.exit(1)

# Send requests
print(f"\nSending {len(test_files)} requests...")
start = time.time()
send_times = {}

requests = {}
for f in test_files:
    if not os.path.exists(f):
        print(f"Skipping missing file: {f}")
        continue

    req_id = str(uuid.uuid4())
    requests[req_id] = f
    request = {'id': req_id, 'path': f}
    req_json = json.dumps(request)
    # Uncomment for debug: print(f"[DEBUG] Sending: {req_json[:100]}...")
    # sys.stdout.flush()
    send_times[req_id] = time.time()  # Track when we sent each request
    process.stdin.write(req_json + '\n')
    process.stdin.flush()

# Wait for results
print("Waiting for results (may take 10-15s per file on Windows)...")
timeout = time.time() + 60  # Increased timeout for Windows
while len(results) < len(requests) and time.time() < timeout:
    time.sleep(0.1)

elapsed = time.time() - start

# Display results
print(f"\n{'='*70}")
print("RESULTS:")
print(f"{'='*70}")

individual_times = []
for req_id, file_path in requests.items():
    filename = os.path.basename(file_path)
    if req_id in results:
        r = results[req_id]
        if r['status'] == 'success':
            # Calculate turnaround time for this file
            if req_id in send_times and '_recv_time' in r:
                turnaround = r['_recv_time'] - send_times[req_id]
                individual_times.append((filename, turnaround))
                print(f"[OK] {filename} ({turnaround:.2f}s)")
                print(f"  Camelot: {r['camelot']} | Open Key: {r['openkey']} | Key: {r['key']}")
            else:
                print(f"[OK] {filename}")
                print(f"  Camelot: {r['camelot']} | Open Key: {r['openkey']} | Key: {r['key']}")
        else:
            print(f"[ERROR] {filename}: {r.get('error', 'Unknown error')}")
    else:
        print(f"[TIMEOUT] {filename}")

print(f"{'='*70}")
print(f"\nProcessed {len(results)}/{len(requests)} files in {elapsed:.2f}s")
if len(results) > 0:
    print(f"Average (wall clock): {elapsed/len(results):.2f}s per file")

# Show individual turnaround times
if individual_times:
    print(f"\nIndividual Processing Times:")
    print(f"{'-'*70}")
    for fname, t in individual_times:
        print(f"  {fname}: {t:.2f}s")
    print(f"{'-'*70}")
    avg_individual = sum(t for _, t in individual_times) / len(individual_times)
    print(f"  Average (per-file): {avg_individual:.2f}s")

# Cleanup
process.stdin.close()
process.wait(timeout=5)
print("\nDone!")
