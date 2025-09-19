const { ipcRenderer } = require("electron");

const messagesDiv = document.getElementById("messages");
const input = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const backBtn = document.getElementById("back-btn");

function addMessage(text, sender) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  msg.innerText = text;
  messagesDiv.appendChild(msg);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

if (backBtn) {
  backBtn.addEventListener("click", () => {
    window.location.href = "../landing.html";
  });
}

function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  addMessage(text, "user");
  input.value = "";

  // Send to backend via Electron IPC
  ipcRenderer.send("user-message", text);
}

// Receive reply from Python
ipcRenderer.on("bot-reply", (event, data) => {
  if (data.reply) {
    addMessage(data.reply, "bot");
  } else if (data.error) {
    addMessage("⚠️ Error: " + data.error, "bot");
  }
});
