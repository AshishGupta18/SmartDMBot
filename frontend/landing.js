document.addEventListener("DOMContentLoaded", () => {
  const go = (file) => {
    window.location.href = file;
  };
  document
    .getElementById("btn-dmbot")
    .addEventListener("click", () => go("dmbot1/index.html"));
  document
    .getElementById("btn-bot2")
    .addEventListener("click", () => go("dmbot2/index.html"));
  document
    .getElementById("btn-bot3")
    .addEventListener("click", () => go("bot3.html"));
});
