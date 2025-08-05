const axios = require('axios');
const chat = document.getElementById('chat');
const questionInput = document.getElementById('question');

async function send() {
  const question = questionInput.value;
  if (!question) return;

  chat.innerHTML += `<div class="user"><b>You:</b> ${question}</div>`;
  questionInput.value = "";

  try {
    const res = await axios.post("http://127.0.0.1:5000/ask", { question });
    chat.innerHTML += `<div class="bot"><b>Bot:</b> ${res.data.answer}</div>`;
  } catch (err) {
    chat.innerHTML += `<div class="bot"><b>Bot:</b> Error connecting to backend</div>`;
  }

  chat.scrollTop = chat.scrollHeight;
}
