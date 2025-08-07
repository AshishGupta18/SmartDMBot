const { app, BrowserWindow, screen } = require('electron');

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
    frame: false, // We'll build our own header bar with minimize/close
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  win.loadFile('index.html');
}

app.whenReady().then(createWindow);