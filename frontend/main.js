const { app, BrowserWindow, screen } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let backendProcess;

function createWindow() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;

  const win = new BrowserWindow({
    width: 400,
    height: 600,
    x: width - 420,
    y: height - 620,
    minWidth: 320,
    minHeight: 500,
    maxWidth: 800,
    maxHeight: 1000,
    resizable: true,
    minimizable: true,
    maximizable: true,
    frame: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  win.loadFile('index.html');
}

function startBackend() {
  // Detect correct path depending on development vs production
  const backendExePath = app.isPackaged
  ? path.join(process.resourcesPath, 'backend', 'backend.exe')
  : path.join(__dirname, '../backend/dist/backend/backend.exe');

  console.log(`Starting backend from: ${backendExePath}`);

  backendProcess = spawn(backendExePath, [], {
    stdio: 'inherit',
    shell: true,
  });

  backendProcess.on('error', (err) => {
    console.error('Failed to start backend:', err);
  });

  backendProcess.on('close', (code) => {
    console.log(`Backend exited with code ${code}`);
  });
}

app.whenReady().then(() => {
  startBackend();
  createWindow();
});

app.on('window-all-closed', () => {
  if (backendProcess) backendProcess.kill(); // kill backend when window closes
  if (process.platform !== 'darwin') app.quit();
});
