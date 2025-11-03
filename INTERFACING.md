# Electron Integration Guide for Musical Key Detection Server

## Overview

The `openkeyscan-analyzer-server` is a long-running Python process that detects musical keys in MP3 files. It communicates via **stdin/stdout using line-delimited JSON (NDJSON)** protocol, making it ideal for Electron IPC integration.

---

## Protocol Specification

### Communication Method
- **Input**: Send JSON requests to the server's **stdin** (one per line)
- **Output**: Receive JSON responses from the server's **stdout** (one per line)
- **Logging**: Server logs debug info to **stderr** (optional to monitor)

### Message Types

#### 1. Request (Electron → Server)
```json
{"id": "unique-uuid-1234", "path": "/absolute/path/to/song.mp3"}
```

**Fields:**
- `id` (string, required): Unique identifier to match responses to requests
- `path` (string, required): Absolute file path to MP3 file (**must not use `~` expansion**)

#### 2. Success Response (Server → Electron)
```json
{
  "id": "unique-uuid-1234",
  "status": "success",
  "camelot": "9A",
  "openkey": "2m",
  "key": "E minor",
  "class_id": 8,
  "filename": "song.mp3"
}
```

**Fields:**
- `id`: Matches the request ID
- `status`: "success"
- `camelot`: Camelot wheel notation (1A-12A minor, 1B-12B major)
- `openkey`: Open Key notation (1m-12m minor, 1d-12d major) for Traktor compatibility
- `key`: Human-readable key name (e.g., "E minor", "C major")
- `class_id`: Neural network output class (0-23)
- `filename`: Name of the analyzed file

#### 3. Error Response (Server → Electron)
```json
{
  "id": "unique-uuid-1234",
  "status": "error",
  "error": "File not found",
  "filename": "song.mp3"
}
```

#### 4. System Messages (Server → Electron)
```json
{"type": "ready"}      // Sent once on startup (model loaded)
{"type": "heartbeat"}  // Sent every 30 seconds (server alive)
```

---

## Implementation Steps

### 1. Spawn the Server Process

```javascript
const { spawn } = require('child_process');
const readline = require('readline');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

class KeyDetectionService {
  constructor(executablePath) {
    this.serverProcess = null;
    this.pendingRequests = new Map(); // Map<requestId, resolver>
    this.executablePath = executablePath;
    this.isReady = false;
  }

  start() {
    return new Promise((resolve, reject) => {
      // Spawn the server process
      this.serverProcess = spawn(this.executablePath, [
        '--workers', '1'  // Adjust workers based on memory constraints
      ]);

      // Set up line reader for stdout (responses)
      const rl = readline.createInterface({
        input: this.serverProcess.stdout,
        crlfDelay: Infinity
      });

      // Handle responses
      rl.on('line', (line) => {
        try {
          const response = JSON.parse(line);
          this.handleResponse(response);
        } catch (err) {
          console.error('Failed to parse server response:', err);
        }
      });

      // Monitor stderr for debugging
      this.serverProcess.stderr.on('data', (data) => {
        console.log('[Server]', data.toString());
      });

      // Handle process exit
      this.serverProcess.on('exit', (code) => {
        console.error(`Server exited with code ${code}`);
        this.isReady = false;
        // Optionally implement auto-restart logic here
      });

      // Wait for ready signal
      const readyTimeout = setTimeout(() => {
        reject(new Error('Server failed to start within 10 seconds'));
      }, 10000);

      this.once('ready', () => {
        clearTimeout(readyTimeout);
        resolve();
      });
    });
  }

  handleResponse(response) {
    // Handle system messages
    if (response.type === 'ready') {
      this.isReady = true;
      this.emit('ready');
      return;
    }

    if (response.type === 'heartbeat') {
      this.emit('heartbeat');
      return;
    }

    // Handle request responses
    if (response.id && this.pendingRequests.has(response.id)) {
      const { resolve, reject, timeout } = this.pendingRequests.get(response.id);
      clearTimeout(timeout);
      this.pendingRequests.delete(response.id);

      if (response.status === 'success') {
        resolve(response);
      } else {
        reject(new Error(response.error || 'Unknown error'));
      }
    }
  }

  analyzeFile(filePath, timeoutMs = 30000) {
    return new Promise((resolve, reject) => {
      if (!this.isReady) {
        return reject(new Error('Server not ready'));
      }

      // Generate unique request ID
      const requestId = uuidv4();

      // Convert to absolute path (important!)
      const absolutePath = path.resolve(filePath);

      // Set up timeout
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        reject(new Error(`Analysis timeout after ${timeoutMs}ms`));
      }, timeoutMs);

      // Store resolver
      this.pendingRequests.set(requestId, { resolve, reject, timeout });

      // Send request
      const request = {
        id: requestId,
        path: absolutePath
      };

      try {
        this.serverProcess.stdin.write(JSON.stringify(request) + '\n');
      } catch (err) {
        this.pendingRequests.delete(requestId);
        clearTimeout(timeout);
        reject(err);
      }
    });
  }

  async analyzeMultiple(filePaths) {
    // Analyze multiple files concurrently
    return Promise.all(
      filePaths.map(filePath => this.analyzeFile(filePath))
    );
  }

  stop() {
    if (this.serverProcess) {
      this.serverProcess.kill();
      this.serverProcess = null;
      this.isReady = false;
    }
  }
}

// Add basic event emitter functionality
Object.assign(KeyDetectionService.prototype, {
  on(event, handler) {
    if (!this._handlers) this._handlers = {};
    if (!this._handlers[event]) this._handlers[event] = [];
    this._handlers[event].push(handler);
  },
  once(event, handler) {
    const onceHandler = (...args) => {
      handler(...args);
      this.off(event, onceHandler);
    };
    this.on(event, onceHandler);
  },
  off(event, handler) {
    if (!this._handlers || !this._handlers[event]) return;
    this._handlers[event] = this._handlers[event].filter(h => h !== handler);
  },
  emit(event, ...args) {
    if (!this._handlers || !this._handlers[event]) return;
    this._handlers[event].forEach(handler => handler(...args));
  }
});
```

### 2. Usage in Electron Main Process

```javascript
const { app } = require('electron');

// Initialize service
const serverPath = app.isPackaged
  ? path.join(process.resourcesPath, 'openkeyscan-analyzer-server')
  : path.join(__dirname, '../dist/openkeyscan-analyzer/openkeyscan-analyzer-server');

const keyService = new KeyDetectionService(serverPath);

// Start server on app ready
app.on('ready', async () => {
  try {
    await keyService.start();
    console.log('Key detection server ready!');

    // Example: Analyze a single file
    const result = await keyService.analyzeFile('/path/to/song.mp3');
    console.log(`Key: ${result.camelot} (${result.key})`);
    console.log(`Open Key: ${result.openkey}`);

    // Example: Analyze multiple files
    const results = await keyService.analyzeMultiple([
      '/path/to/song1.mp3',
      '/path/to/song2.mp3',
      '/path/to/song3.mp3'
    ]);

    results.forEach(r => {
      console.log(`${r.filename}: ${r.camelot} (${r.key})`);
    });

  } catch (err) {
    console.error('Failed to start key detection server:', err);
  }
});

// Clean up on quit
app.on('before-quit', () => {
  keyService.stop();
});
```

### 3. Expose to Renderer via IPC (Optional)

```javascript
const { ipcMain } = require('electron');

// Register IPC handler
ipcMain.handle('analyze-key', async (event, filePath) => {
  try {
    const result = await keyService.analyzeFile(filePath);
    return { success: true, data: result };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// In renderer process:
// const result = await ipcRenderer.invoke('analyze-key', '/path/to/song.mp3');
```

---

## Important Implementation Notes

### ✅ Path Handling
- **ALWAYS use absolute paths** - The server doesn't expand `~` or relative paths
- Convert paths: `path.resolve(filePath)` or `path.join(app.getPath('music'), 'song.mp3')`
- ❌ Bad: `~/Music/song.mp3`
- ✅ Good: `/Users/username/Music/song.mp3`

### ✅ Request ID Management
- Use UUIDs or unique identifiers for each request
- Store pending requests in a Map to match responses
- Implement timeouts (recommend 30s per file)

### ✅ Error Handling
- Handle JSON parse errors (malformed responses)
- Handle process exit/crash (implement auto-restart)
- Handle timeouts (slow files or stuck processing)
- Reject pending requests on process exit

### ✅ Memory Management
- Default: 1 worker thread (~1GB peak memory for batch processing)
- Increase workers for higher throughput: `--workers 2` (~1.2GB peak)
- Monitor memory usage if processing large batches

### ✅ Process Lifecycle
- Start server on app ready (wait for `{"type": "ready"}`)
- Keep server running for app lifetime
- Stop server on app quit
- Consider auto-restart on crash with exponential backoff

---

## Performance Characteristics

### Throughput
- **Single file**: ~440ms average (includes audio loading + inference)
- **Concurrent (10 files)**: ~437ms average per file
- **Expected**: 130-150 files/minute with 1 worker

### Memory Usage
- **Baseline**: ~200MB (model loaded)
- **Peak (1 worker)**: ~1.1GB (10 files concurrent)
- **Peak (2 workers)**: ~1.2GB (10 files concurrent)
- **Peak (4 workers)**: ~1.6GB (estimated)

### Startup Time
- **Model loading**: ~1-2 seconds
- **Ready signal**: ~1.5-2.5 seconds total

---

## Troubleshooting

### Server doesn't start
- Check executable path exists
- Check executable permissions (`chmod +x`)
- Monitor stderr for error messages

### "File not found" errors
- Verify absolute paths (not relative or `~`)
- Check file actually exists and is readable
- Verify file extension is `.mp3`

### Timeout errors
- Increase timeout for large files
- Check server process is still running
- Monitor memory usage (may be OOM)

### Memory crashes
- Reduce workers to 1
- Process files in smaller batches
- Add delays between batch processing

---

## Advanced: Auto-Restart Logic

```javascript
class RobustKeyDetectionService extends KeyDetectionService {
  constructor(executablePath, maxRetries = 3) {
    super(executablePath);
    this.maxRetries = maxRetries;
    this.retryCount = 0;
    this.retryDelay = 1000; // Start with 1s
  }

  async start() {
    while (this.retryCount < this.maxRetries) {
      try {
        await super.start();
        this.retryCount = 0; // Reset on success
        this.retryDelay = 1000;
        return;
      } catch (err) {
        this.retryCount++;
        console.error(`Server start failed (attempt ${this.retryCount}):`, err);

        if (this.retryCount >= this.maxRetries) {
          throw new Error(`Server failed to start after ${this.maxRetries} attempts`);
        }

        // Exponential backoff
        await new Promise(resolve => setTimeout(resolve, this.retryDelay));
        this.retryDelay *= 2;
      }
    }
  }

  handleProcessExit(code) {
    console.error(`Server crashed with code ${code}, restarting...`);

    // Reject all pending requests
    for (const [id, { reject, timeout }] of this.pendingRequests) {
      clearTimeout(timeout);
      reject(new Error('Server crashed'));
    }
    this.pendingRequests.clear();

    // Auto-restart
    setTimeout(() => this.start(), 2000);
  }
}
```

---

## Testing Checklist

- [ ] Server starts successfully and sends `{"type": "ready"}`
- [ ] Single file analysis returns correct format
- [ ] Multiple concurrent files process correctly
- [ ] Absolute paths work correctly
- [ ] Error responses received for invalid files
- [ ] Timeout handling works
- [ ] Process exit/crash handled gracefully
- [ ] Memory usage stays within limits during batch processing
- [ ] Server can be stopped cleanly on app quit

---

## Example Output

```javascript
// Success result
{
  id: 'abc-123',
  status: 'success',
  camelot: '9A',      // Camelot notation (DJ mixing)
  openkey: '2m',      // Open Key notation (Traktor)
  key: 'E minor',     // Human-readable
  class_id: 8,        // Neural network class
  filename: 'song.mp3'
}

// Error result
{
  id: 'abc-123',
  status: 'error',
  error: 'File not found',
  filename: 'song.mp3'
}
```

---

## Summary

**Key Points for Electron Integration:**
1. Spawn `openkeyscan-analyzer-server` as child process
2. Use `readline` to parse line-delimited JSON from stdout
3. Send requests to stdin (one JSON per line)
4. Use UUIDs to match async responses to requests
5. Always use absolute paths
6. Implement timeouts and error handling
7. Monitor server health via heartbeats
8. Clean up process on app quit

This architecture provides **low-latency**, **high-throughput** key detection with **simple IPC** and **automatic concurrency** handling.
