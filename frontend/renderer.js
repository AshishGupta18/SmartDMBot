const chat = document.getElementById('chat');
const questionInput = document.getElementById('question');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');

async function typeText(element, text, delay = 20) {
  element.innerHTML = "";
  const words = text.split(" ");
  for (let i = 0; i < words.length; i++) {
    element.innerHTML += words[i];
    if (i !== words.length - 1) element.innerHTML += " ";
    await new Promise(resolve => setTimeout(resolve, delay));
    chat.scrollTop = chat.scrollHeight;
  }
}

async function send() {
  const question = questionInput.value.trim();
  if (!question) return;

  chat.innerHTML += `<div class="user"><b>You:</b> ${question}</div>`;
  questionInput.value = "";

  const typingDiv = document.createElement("div");
  typingDiv.className = "bot";
  typingDiv.textContent = "Smart DMBot is typing...";
  chat.appendChild(typingDiv);
  chat.scrollTop = chat.scrollHeight;

  try {
    const res = await axios.post("http://127.0.0.1:5000/ask", { question });
    typingDiv.innerHTML = "";
    await typeText(typingDiv, `🤖 ${res.data.answer}`, 25);
  } catch (err) {
    typingDiv.innerHTML = "Error connecting to backend.";
  }

  chat.scrollTop = chat.scrollHeight;
}

document.querySelector(".close-btn").addEventListener("click", () => {
  const window = remote.getCurrentWindow();
  window.close();
});

sendBtn.addEventListener("click", send);
questionInput.addEventListener("keydown", e => {
  if (e.key === "Enter") send();
});

micBtn.addEventListener("click", () => {
  const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.lang = "en-US";
  recognition.interimResults = false;

  recognition.onresult = (event) => {
    const spokenText = event.results[0][0].transcript;
    questionInput.value = spokenText;
    send();
  };

  recognition.onerror = (event) => {
    alert("Speech recognition error: " + event.error);
  };

  recognition.start();
});

const { remote } = require('electron');
const win = remote.getCurrentWindow();

document.querySelector(".minimize").addEventListener("click", () => {
  win.minimize();
});

document.querySelector(".close").addEventListener("click", () => {
  win.close();
});