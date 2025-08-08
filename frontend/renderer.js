const chat = document.getElementById("chat");
const questionInput = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-btn");

// Function to download images
async function downloadImage(imageUrl, filename) {
  try {
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (error) {
    console.error("Error downloading image:", error);
    alert("Failed to download image. Please try again.");
  }
}

// Function to check if image loads successfully
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      // Additional validation: check if image has actual dimensions
      if (img.naturalWidth === 0 || img.naturalHeight === 0) {
        reject(new Error("Image has no dimensions"));
      } else {
        resolve(img);
      }
    };
    img.onerror = () => reject(new Error("Image failed to load"));
    img.src = src;
  });
}

// Function to validate if image URL actually returns an image
async function validateImageUrl(url) {
  try {
    const response = await fetch(url, { method: "HEAD" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("image/")) {
      throw new Error("URL does not return an image");
    }
    return true;
  } catch (error) {
    throw new Error(`Image validation failed: ${error.message}`);
  }
}

async function typeText(element, text, delay = 20) {
  element.innerHTML = "";

  // First, process the text to convert **text** to <b>text</b>
  let processedText = text.replace(/\*\*(.*?)\*\*/g, "<b>$1</b>");

  // Split by words but preserve HTML tags
  const words = processedText.split(/(\s+)/);

  for (let i = 0; i < words.length; i++) {
    element.innerHTML += words[i];
    if (i !== words.length - 1) {
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
    chat.scrollTop = chat.scrollHeight;
  }
}

async function send() {
  const question = questionInput.value.trim();
  if (!question) return;

  // Hide placeholder if present
  const placeholder = document.getElementById("chat-placeholder");
  if (placeholder) placeholder.style.display = "none";

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

    // Only try to display image if SVG path is provided
    if (res.data.svg) {
      // Add timestamp to prevent browser caching
      const timestamp = new Date().getTime();
      const randomId = Math.random().toString(36).substring(7);
      const imageUrl = `http://127.0.0.1:5000${res.data.svg}?t=${timestamp}&r=${randomId}`;

      try {
        // First validate that the URL actually returns an image
        await validateImageUrl(imageUrl);

        // Then try to load the image
        await loadImage(imageUrl);

        // If both validations pass, display it with download option
        const imageContainer = document.createElement("div");
        imageContainer.style.marginTop = "12px";

        const svgImg = document.createElement("img");
        svgImg.src = imageUrl;
        svgImg.alt = "Generated diagram";
        svgImg.style.display = "block";
        svgImg.style.maxWidth = "100%";
        svgImg.style.cursor = "pointer";
        svgImg.title = "Click to download";

        // Make image clickable to download
        svgImg.addEventListener("click", () => {
          downloadImage(svgImg.src, "diagram.svg");
        });

        // Create download button
        const downloadBtn = document.createElement("button");
        downloadBtn.textContent = "📥 Download Diagram";
        downloadBtn.style.marginTop = "8px";
        downloadBtn.style.padding = "6px 12px";
        downloadBtn.style.backgroundColor = "#1a73e8";
        downloadBtn.style.color = "white";
        downloadBtn.style.border = "none";
        downloadBtn.style.borderRadius = "4px";
        downloadBtn.style.cursor = "pointer";
        downloadBtn.style.fontSize = "12px";

        downloadBtn.addEventListener("click", () => {
          downloadImage(svgImg.src, "diagram.svg");
        });

        imageContainer.appendChild(svgImg);
        imageContainer.appendChild(downloadBtn);
        typingDiv.appendChild(imageContainer);

        console.log("Image successfully loaded and displayed");
      } catch (imageError) {
        // If image fails to load, don't display anything
        console.log(
          "Image not available or failed to load:",
          imageError.message
        );
      }
    } else {
      // No diagram generated for this response - but don't clear previous images
      console.log("No diagram generated for this response");
    }
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
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});

micBtn.addEventListener("click", () => {
  const recognition = new (window.SpeechRecognition ||
    window.webkitSpeechRecognition)();
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

const { remote } = require("electron");
const win = remote.getCurrentWindow();

document.querySelector(".minimize").addEventListener("click", () => {
  win.minimize();
});

document.querySelector(".close").addEventListener("click", () => {
  win.close();
});
