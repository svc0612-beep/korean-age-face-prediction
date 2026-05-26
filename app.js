const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureBtn = document.getElementById("captureBtn");
const retryBtn = document.getElementById("retryBtn");
const cameraStatus = document.getElementById("cameraStatus");
const placeholder = document.getElementById("placeholder");
const resultAge = document.getElementById("resultAge");
const resultRange = document.getElementById("resultRange");
const modelMae = document.getElementById("modelMae");
const elapsed = document.getElementById("elapsed");
const analysisList = document.getElementById("analysisList");
const webcamCorrection = document.getElementById("webcamCorrection");

let stream = null;
let faceDetector = null;
const FACE_CROP_SCALE = 1.70;
const FACE_CROP_MAX_RATIO = 0.56;
const FALLBACK_CROP_RATIO = 0.40;
const FALLBACK_CENTER_X = 0.56;
const FALLBACK_CENTER_Y = 0.43;

function setStatus(message, type = "") {
  cameraStatus.textContent = message;
  cameraStatus.className = `status ${type}`.trim();
}

function stopCamera() {
  if (!stream) return;
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  setStatus("\ucd2c\uc601 \uc644\ub8cc", "ready");
}

async function startCamera() {
  try {
    if (stream) stopCamera();
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    setStatus("\uce74\uba54\ub77c \ud65c\uc131\ud654", "ready");
  } catch (error) {
    setStatus("\uce74\uba54\ub77c \uad8c\ud55c \ud544\uc694", "error");
    resultRange.textContent = "\ube0c\ub77c\uc6b0\uc800\uc5d0\uc11c \uce74\uba54\ub77c \uad8c\ud55c\uc744 \ud5c8\uc6a9\ud574\uc57c \ucd2c\uc601\ud560 \uc218 \uc788\uc2b5\ub2c8\ub2e4.";
  }
}

function clampCropBox(box, videoWidth, videoHeight) {
  const side = Math.min(Math.max(box.width, box.height) * FACE_CROP_SCALE, Math.min(videoWidth, videoHeight) * FACE_CROP_MAX_RATIO);
  const centerX = box.x + box.width / 2;
  const centerY = box.y + box.height / 2;
  const sx = Math.min(Math.max(0, centerX - side / 2), videoWidth - side);
  const sy = Math.min(Math.max(0, centerY - side / 2), videoHeight - side);
  return { sx, sy, side };
}

async function detectFaceCropBox() {
  if (!("FaceDetector" in window)) return null;

  try {
    if (!faceDetector) {
      faceDetector = new FaceDetector({ maxDetectedFaces: 1, fastMode: true });
    }

    const probe = document.createElement("canvas");
    probe.width = video.videoWidth;
    probe.height = video.videoHeight;
    probe.getContext("2d").drawImage(video, 0, 0, probe.width, probe.height);
    const faces = await faceDetector.detect(probe);
    if (!faces.length) return null;

    const face = faces.sort((a, b) => {
      const areaA = a.boundingBox.width * a.boundingBox.height;
      const areaB = b.boundingBox.width * b.boundingBox.height;
      return areaB - areaA;
    })[0];
    return clampCropBox(face.boundingBox, video.videoWidth, video.videoHeight);
  } catch (_) {
    return null;
  }
}

function fallbackGuideCropBox() {
  const sourceSide = Math.min(video.videoWidth, video.videoHeight) * FALLBACK_CROP_RATIO;
  const centerX = video.videoWidth * FALLBACK_CENTER_X;
  const centerY = video.videoHeight * FALLBACK_CENTER_Y;
  return {
    sx: Math.min(Math.max(0, centerX - sourceSide / 2), video.videoWidth - sourceSide),
    sy: Math.min(Math.max(0, centerY - sourceSide / 2), video.videoHeight - sourceSide),
    side: sourceSide,
  };
}

async function drawCapture() {
  const context = canvas.getContext("2d");
  const cropBox = (await detectFaceCropBox()) || fallbackGuideCropBox();
  canvas.width = 512;
  canvas.height = 512;
  context.drawImage(video, cropBox.sx, cropBox.sy, cropBox.side, cropBox.side, 0, 0, canvas.width, canvas.height);
  placeholder.style.display = "none";
  return canvas.toDataURL("image/jpeg", 0.92);
}

function renderAnalysis(items) {
  analysisList.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    analysisList.appendChild(li);
  });
}

async function predict(imageData) {
  captureBtn.disabled = true;
  captureBtn.textContent = "\ubd84\uc11d \uc911";
  resultRange.textContent = "\ub85c\uceec PyTorch \ubaa8\ub378\uc774 \uc774\ubbf8\uc9c0\ub97c \ubd84\uc11d\ud558\uace0 \uc788\uc2b5\ub2c8\ub2e4.";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: imageData,
        already_cropped: true,
        apply_webcam_correction: webcamCorrection.checked,
      }),
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "prediction failed");

    resultAge.textContent = `${data.rounded_age}\uc138`;
    const correctionText = data.webcam_offset
      ? ` / \uc6f9\ucea0 \ubcf4\uc815 ${data.webcam_offset > 0 ? "+" : ""}${data.webcam_offset}\uc138`
      : "";
    const reliabilityText = data.low_webcam_reliability
      ? " / \uc8fc\uc758: \uc6f9\ucea0\uc5d0\uc11c \uc800\uc5f0\ub839 \uc3e0\ub9bc \uac10\uc9c0"
      : "";
    resultRange.textContent = `\ubcf4\uc815 \uc804 \ubaa8\ub378\uac12 ${data.raw_age}\uc138, \ucd5c\uc885 \uc608\uce21 ${data.predicted_age}\uc138, \ucc38\uace0 \ubc94\uc704 ${data.age_range[0]}~${data.age_range[1]}\uc138${correctionText}${reliabilityText}`;
    modelMae.textContent = `\u00b1${data.model_mae}\uc138`;
    elapsed.textContent = `${data.elapsed_ms}ms`;
    renderAnalysis(data.analysis);
  } catch (error) {
    resultAge.textContent = `--\uc138`;
    resultRange.textContent = `\ubd84\uc11d \uc2e4\ud328: ${error.message}`;
    renderAnalysis([
      "\uc11c\ubc84\uac00 \uc2e4\ud589 \uc911\uc778\uc9c0 \ud655\uc778\ud558\uc138\uc694.",
      "\ubaa8\ub378 \ud30c\uc77c \uacbd\ub85c\uc640 PyTorch \uc124\uce58 \uc0c1\ud0dc\ub97c \ud655\uc778\ud558\uc138\uc694.",
    ]);
  } finally {
    captureBtn.disabled = false;
    captureBtn.textContent = "\ucc30\uce75";
  }
}

captureBtn.addEventListener("click", async () => {
  if (!video.videoWidth || !stream) {
    resultRange.textContent = "\uce74\uba54\ub77c\uac00 \uc544\uc9c1 \uc900\ube44\ub418\uc9c0 \uc54a\uc558\uc2b5\ub2c8\ub2e4.";
    return;
  }
  const imageData = await drawCapture();
  stopCamera();
  await predict(imageData);
});

retryBtn.addEventListener("click", startCamera);

startCamera();
