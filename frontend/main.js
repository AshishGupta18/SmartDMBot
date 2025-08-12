const { app, BrowserWindow, screen } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let backendProcess = null;

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
      contextIsolation: false
    }
  });

  win.loadFile('index.html');
}

app.whenReady().then(() => {
  // Backend EXE location (inside packaged app)
  const exePath = path.join(__dirname, 'backend-bin', 'MyBackendApp.exe');

  backendProcess = spawn(exePath, [], { shell: true });

  backendProcess.stdout.on('data', (data) => {
    console.log(`[Backend]: ${data}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`[Backend ERROR]: ${data}`);
  });

  backendProcess.on('close', (code) => {
    console.log(`Backend process closed with code ${code}`);
  });

  createWindow();
});

app.on('window-all-closed', () => {
  if (backendProcess) backendProcess.kill();
  app.quit();
});
