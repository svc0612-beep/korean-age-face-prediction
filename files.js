const uploadForm = document.getElementById("uploadForm");
const uploadStatus = document.getElementById("uploadStatus");
const fileList = document.getElementById("fileList");
const refreshFiles = document.getElementById("refreshFiles");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderPreview(file) {
  if (file.preview_type === "image") {
    return `<div class="preview"><img src="${file.download_url}" alt="${escapeHtml(file.title)}" /></div>`;
  }
  if (file.preview_text) {
    return `<div class="preview">${escapeHtml(file.preview_text)}</div>`;
  }
  if (file.preview_type === "pdf") {
    return `<div class="preview file-preview-note">PDF \ubb38\uc11c\ub294 \ud398\uc774\uc9c0 \ub0b4 \uc911\ubcf5 \ub2e4\uc6b4\ub85c\ub4dc \ubc84\ud2bc\uc744 \ud53c\ud558\uae30 \uc704\ud574 \ubbf8\ub9ac\ubcf4\uae30\ub97c \uc694\uc57d\uc73c\ub85c\ub9cc \ud45c\uc2dc\ud569\ub2c8\ub2e4.</div>`;
  }
  return "";
}

function renderFiles(files) {
  const visibleFiles = dedupeFiles(files);
  if (!visibleFiles.length) {
    fileList.innerHTML = `<article class="file-card"><div><h3>\uc544\uc9c1 \uc5c5\ub85c\ub4dc\ub41c \uc790\ub8cc\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.</h3><p class="muted">\ubc1c\ud45c \uc790\ub8cc\ub97c \uba3c\uc800 \uc5c5\ub85c\ub4dc\ud558\uc138\uc694.</p></div></article>`;
    return;
  }

  fileList.innerHTML = visibleFiles
    .map(
      (file) => `
        <article class="file-card">
          <div>
            <h3>${escapeHtml(file.title || file.original_name)}</h3>
            <p class="muted">${escapeHtml(file.description || "\uc124\uba85 \uc5c6\uc74c")}</p>
            <div class="file-meta">${escapeHtml(file.original_name)} / ${escapeHtml(file.size_label)} / ${escapeHtml(file.uploaded_at)}</div>
            ${renderPreview(file)}
          </div>
          <a class="secondary-btn" href="${file.download_url}" download>\ub2e4\uc6b4\ub85c\ub4dc</a>
        </article>
      `
    )
    .join("");
}

function dedupeFiles(files) {
  const seen = new Set();
  return [...files]
    .sort((a, b) => String(b.uploaded_at || "").localeCompare(String(a.uploaded_at || "")))
    .filter((file) => {
      const key = `${file.original_name || file.stored_name}:${file.size || 0}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

async function loadFiles() {
  const response = await fetch("/api/uploads");
  const data = await response.json();
  renderFiles(data.files || []);
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(uploadForm);
  uploadStatus.textContent = "\uc5c5\ub85c\ub4dc \uc911...";

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "upload failed");
    uploadStatus.textContent = "\uc5c5\ub85c\ub4dc \uc644\ub8cc";
    uploadForm.reset();
    await loadFiles();
  } catch (error) {
    uploadStatus.textContent = `\uc5c5\ub85c\ub4dc \uc2e4\ud328: ${error.message}`;
  }
});

refreshFiles.addEventListener("click", loadFiles);

loadFiles();
