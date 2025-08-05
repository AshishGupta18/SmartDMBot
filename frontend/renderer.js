const chat = document.getElementById('chat');
const questionInput = document.getElementById('question');
const sendBtn = document.getElementById('send-btn');

async function typeText(element, text, delay = 50) {
  element.innerHTML = "";  // Clear existing content
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

  // Show user question immediately
  chat.innerHTML += `<div class="user"><b>You:</b> ${question}</div>`;
  questionInput.value = "";

  // Show typing indicator
  const typingDiv = document.createElement("div");
  typingDiv.className = "bot";
  typingDiv.id = "typing";
  typingDiv.innerHTML = "Smart DMBot is typing...";
  chat.appendChild(typingDiv);
  chat.scrollTop = chat.scrollHeight;

  try {
    const res = await axios.post("http://127.0.0.1:5000/ask", { question });
    typingDiv.innerHTML = "";  // Clear "typing" text
    
    // Gradually type the bot's answer word by word
    await typeText(typingDiv, `<b>Bot:</b> ${res.data.answer}`, 50);
    
  } catch (err) {
    typingDiv.innerHTML = "<b>Bot:</b> Error connecting to backend";
  }

  chat.scrollTop = chat.scrollHeight;
}

sendBtn.addEventListener("click", send);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});