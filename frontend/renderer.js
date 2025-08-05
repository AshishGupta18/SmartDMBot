const chat = document.getElementById('chat');
const questionInput = document.getElementById('question');
const sendBtn = document.getElementById('send-btn');

// Create floating scroll button
const scrollButton = document.createElement("button");
scrollButton.textContent = "⬇ Scroll to Bottom";
scrollButton.id = "scroll-btn";
scrollButton.style.position = "fixed";
scrollButton.style.bottom = "90px";
scrollButton.style.right = "20px";
scrollButton.style.padding = "10px 15px";
scrollButton.style.background = "linear-gradient(135deg, #42a5f5, #478ed1)";
scrollButton.style.color = "#fff";
scrollButton.style.border = "none";
scrollButton.style.borderRadius = "20px";
scrollButton.style.cursor = "pointer";
scrollButton.style.display = "none";
scrollButton.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.3)";
scrollButton.style.zIndex = "1000";
document.body.appendChild(scrollButton);

scrollButton.addEventListener("click", () => {
  chat.scrollTop = chat.scrollHeight;
});

// Show/hide scroll button on scroll
chat.addEventListener("scroll", () => {
  if (chat.scrollHeight - chat.scrollTop > chat.clientHeight + 100) {
    scrollButton.style.display = "block";
  } else {
    scrollButton.style.display = "none";
  }
});

// Auto-resize textarea dynamically
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = questionInput.scrollHeight + "px";
});

// Allow Enter to send and Shift+Enter for new line
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// Function to copy text to clipboard
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    alert("Copied to clipboard!");
  }).catch(() => {
    alert("Failed to copy text.");
  });
}

// Function to animate bot reply word by word
async function typeText(element, text, delay = 50) {
  element.innerHTML = "";
  const words = text.split(" ");
  for (let i = 0; i < words.length; i++) {
    element.innerHTML += words[i] + " ";
    await new Promise(resolve => setTimeout(resolve, delay));
    chat.scrollTop = chat.scrollHeight;
  }
}

async function send() {
  const question = questionInput.value.trim();
  if (!question) return;

  // Reset textarea height after sending
  questionInput.style.height = "40px";

  // Show user question
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
    typingDiv.innerHTML = "";

    // Create bot message container
    const botMessageDiv = document.createElement("div");
    botMessageDiv.className = "bot";

    // Create header with "Copy" button
    const headerDiv = document.createElement("div");
    headerDiv.style.display = "flex";
    headerDiv.style.justifyContent = "space-between";
    headerDiv.style.alignItems = "center";
    headerDiv.style.marginBottom = "5px";

    const copyText = document.createElement("span");
    copyText.textContent = "Copy";
    copyText.style.marginRight = "10px";
    copyText.style.fontSize = "12px";
    copyText.style.cursor = "pointer";
    copyText.style.color = "#fff";

    const copyButton = document.createElement("button");
    copyButton.textContent = "📋";
    copyButton.style.background = "transparent";
    copyButton.style.border = "none";
    copyButton.style.cursor = "pointer";
    copyButton.style.fontSize = "14px";
    copyButton.style.color = "#fff";

    copyButton.addEventListener("click", () => {
      copyToClipboard(res.data.answer);
    });
    copyText.addEventListener("click", () => {
      copyToClipboard(res.data.answer);
    });

    headerDiv.appendChild(copyText);
    headerDiv.appendChild(copyButton);

    // Create message text area
    const messageTextDiv = document.createElement("div");
    botMessageDiv.appendChild(headerDiv);
    botMessageDiv.appendChild(messageTextDiv);
    chat.appendChild(botMessageDiv);

    // Type the text gradually
    await typeText(messageTextDiv, `Bot: ${res.data.answer}`, 50);

  } catch (err) {
    typingDiv.innerHTML = "<b>Bot:</b> Error connecting to backend";
  }

  chat.scrollTop = chat.scrollHeight;
}

sendBtn.addEventListener("click", send);