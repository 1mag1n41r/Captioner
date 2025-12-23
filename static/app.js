(function () {
  const body = document.body;

  // ---- Theme (dark mode) ----
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme === "dark") body.classList.add("dark");

  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      body.classList.toggle("dark");
      localStorage.setItem("theme", body.classList.contains("dark") ? "dark" : "light");
    });
  }

  // ---- File upload label ----
  const fileInput = document.getElementById("fileInput");
  const fileLabel = document.getElementById("fileLabel");
  if (fileInput && fileLabel) {
    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0) {
        fileLabel.textContent = fileInput.files[0].name;
      }
    });
  }

  // ---- Loading overlay ----
  const overlay = document.getElementById("loadingOverlay");
  function showLoading() {
    if (overlay) overlay.classList.remove("hidden");
  }

  const uploadForm = document.getElementById("uploadForm");
  if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
      showLoading();
    });
  }

  const captionForm = document.getElementById("captionForm");
  if (captionForm) {
    captionForm.addEventListener("submit", () => {
      showLoading();
    });
  }

  // ---- Copy caption ----
  const copyBtn = document.getElementById("copyBtn");
  const captionText = document.getElementById("captionText");
  if (copyBtn && captionText) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(captionText.textContent.trim());
        copyBtn.textContent = "Copied!";
        setTimeout(() => (copyBtn.textContent = "Copy caption"), 1200);
      } catch {
        copyBtn.textContent = "Copy failed";
        setTimeout(() => (copyBtn.textContent = "Copy caption"), 1200);
      }
    });
  }
})();
