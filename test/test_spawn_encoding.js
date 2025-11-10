#!/usr/bin/env node

/**
 * Test Node.js spawn with special characters in filenames
 * Replicates the Electron app's key-detection-service.ts to test for UTF-8/UTF-16 encoding issues
 */

const { spawn } = require('child_process');
const readline = require('readline');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Configuration
const MUSIC_DIR = path.join(os.homedir(), 'Music', 'accents');
const EXE_PATH_MAC = path.join(__dirname, '../dist/openkeyscan-analyzer/openkeyscan-analyzer');
const EXE_PATH_WIN = path.join(__dirname, '../dist/openkeyscan-analyzer/openkeyscan-analyzer.exe');
const WORKER_COUNT = 1;

// Determine executable path
const isWindows = process.platform === 'win32';
const executablePath = isWindows ? EXE_PATH_WIN : EXE_PATH_MAC;

// Colors for output (ANSI escape codes)
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  gray: '\x1b[90m',
};

function log(message, color = colors.reset) {
  console.log(color + message + colors.reset);
}

// Find audio files
function findAudioFiles(dir) {
  if (!fs.existsSync(dir)) {
    log(`Error: Directory not found: ${dir}`, colors.red);
    return [];
  }

  const extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.mp4', '.aiff', '.au'];
  const files = fs.readdirSync(dir)
    .filter(file => extensions.includes(path.extname(file).toLowerCase()))
    .map(file => path.join(dir, file));

  return files;
}

// Main test function
async function runTest() {
  log('='.repeat(80), colors.cyan);
  log('Testing Node.js spawn with special character filenames', colors.cyan);
  log('='.repeat(80), colors.cyan);
  console.log();

  // Check if executable exists
  if (!fs.existsSync(executablePath)) {
    log(`Error: Executable not found at: ${executablePath}`, colors.red);
    log('Please build the executable first with: pyinstaller openkeyscan_analyzer.spec', colors.yellow);
    process.exit(1);
  }

  // Find test files
  const audioFiles = findAudioFiles(MUSIC_DIR);
  if (audioFiles.length === 0) {
    log(`Error: No audio files found in ${MUSIC_DIR}`, colors.red);
    process.exit(1);
  }

  log(`Found ${audioFiles.length} audio files to test:`, colors.cyan);
  audioFiles.forEach((file, idx) => {
    const basename = path.basename(file);
    log(`  ${idx + 1}. ${basename}`, colors.gray);
  });
  console.log();

  // Print encoding information
  log('Environment Information:', colors.cyan);
  log(`  Platform: ${process.platform}`, colors.gray);
  log(`  Node.js version: ${process.version}`, colors.gray);
  log(`  Default encoding: ${process.stdout.encoding || 'utf8'}`, colors.gray);
  log(`  Executable: ${executablePath}`, colors.gray);
  log(`  Workers: ${WORKER_COUNT}`, colors.gray);
  console.log();

  // Start the server
  log('Starting key detection server...', colors.cyan);

  const serverProcess = spawn(executablePath, ['--workers', WORKER_COUNT.toString()], {
    stdio: ['pipe', 'pipe', 'pipe'],
    // Explicitly set encoding to utf8 for all streams
    env: { ...process.env }
  });

  if (!serverProcess.stdout || !serverProcess.stderr || !serverProcess.stdin) {
    log('Error: Failed to create server process streams', colors.red);
    process.exit(1);
  }

  // Set encoding explicitly
  serverProcess.stdin.setDefaultEncoding('utf8');
  serverProcess.stdout.setEncoding('utf8');
  serverProcess.stderr.setEncoding('utf8');

  // Set up line reader for stdout
  const rl = readline.createInterface({
    input: serverProcess.stdout,
    crlfDelay: Infinity,
  });

  // Track results
  const pendingRequests = new Map();
  const results = [];
  let isReady = false;

  // Handle responses
  rl.on('line', (line) => {
    try {
      const response = JSON.parse(line);

      // Handle system messages
      if (response.type === 'ready') {
        log('[OK] Server is ready!', colors.green);
        isReady = true;
        return;
      }

      if (response.type === 'heartbeat') {
        log('[DEBUG] Heartbeat received', colors.gray);
        return;
      }

      // Handle analysis responses
      if (response.id && pendingRequests.has(response.id)) {
        const { resolve } = pendingRequests.get(response.id);
        pendingRequests.delete(response.id);
        results.push(response);
        resolve(response);
      }
    } catch (err) {
      // Non-JSON output from subprocess
      log(`[SUBPROCESS] ${line}`, colors.gray);
    }
  });

  // Monitor stderr
  serverProcess.stderr.on('data', (data) => {
    const message = data.toString().trim();
    if (message) {
      log(`[STDERR] ${message}`, colors.yellow);
    }
  });

  // Handle process exit
  serverProcess.on('exit', (code) => {
    log(`\n[EXIT] Server exited with code ${code}`, code === 0 ? colors.gray : colors.red);
  });

  // Handle process errors
  serverProcess.on('error', (err) => {
    log(`[ERROR] Server process error: ${err.message}`, colors.red);
  });

  // Wait for ready signal
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Server failed to start within 30 seconds'));
    }, 30000);

    const checkReady = setInterval(() => {
      if (isReady) {
        clearTimeout(timeout);
        clearInterval(checkReady);
        resolve();
      }
    }, 100);
  });

  console.log();
  log('Sending analysis requests...', colors.cyan);
  console.log();

  const startTime = Date.now();

  // Send requests for all files
  const promises = audioFiles.map((filePath, idx) => {
    return new Promise((resolve, reject) => {
      const requestId = `req-${idx}`;
      pendingRequests.set(requestId, { resolve, reject });

      const request = {
        id: requestId,
        path: filePath,
      };

      // Log what we're sending (with encoding info)
      const jsonStr = JSON.stringify(request);
      const basename = path.basename(filePath);
      log(`[SEND] ${basename}`, colors.gray);
      log(`  Path: ${filePath}`, colors.gray);
      log(`  JSON length: ${jsonStr.length} bytes`, colors.gray);
      log(`  Buffer length: ${Buffer.from(jsonStr, 'utf8').length} bytes`, colors.gray);

      try {
        serverProcess.stdin.write(jsonStr + '\n', 'utf8');
      } catch (err) {
        pendingRequests.delete(requestId);
        reject(err);
      }
    });
  });

  // Wait for all responses
  await Promise.allSettled(promises);

  const elapsed = Date.now() - startTime;

  // Print results
  console.log();
  log('='.repeat(80), colors.cyan);
  log('RESULTS:', colors.cyan);
  log('='.repeat(80), colors.cyan);
  console.log();

  let successCount = 0;
  let failedCount = 0;

  results.forEach((result) => {
    const basename = path.basename(result.filename || 'unknown');

    if (result.status === 'success') {
      log(`[OK] ${basename}`, colors.green);
      log(`  Camelot: ${result.camelot} | Open Key: ${result.openkey} | Key: ${result.key}`, colors.gray);
      successCount++;
    } else {
      log(`[FAIL] ${basename}`, colors.red);
      log(`  Error: ${result.error}`, colors.yellow);
      failedCount++;
    }
    console.log();
  });

  log('='.repeat(80), colors.cyan);
  log(`Processed ${results.length}/${audioFiles.length} files in ${(elapsed / 1000).toFixed(2)}s`, colors.cyan);
  if (audioFiles.length > 0) {
    log(`Average: ${(elapsed / audioFiles.length / 1000).toFixed(2)}s per file`, colors.cyan);
  }
  log(`Success: ${successCount}, Failed: ${failedCount}`, successCount === audioFiles.length ? colors.green : colors.yellow);
  log('='.repeat(80), colors.cyan);

  // Cleanup
  serverProcess.stdin.end();
  serverProcess.kill();

  // Exit with appropriate code
  process.exit(failedCount > 0 ? 1 : 0);
}

// Run the test
runTest().catch((err) => {
  log(`\nFatal error: ${err.message}`, colors.red);
  console.error(err);
  process.exit(1);
});
