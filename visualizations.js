const slider = document.getElementById("vizSize");
const grid = document.querySelector(".viz-page-grid");

function applySize() {
  const value = Number(slider.value);
  grid.style.setProperty("--viz-zoom", String(value / 100));
  grid.style.setProperty("--caption-size", `${Math.round(13 + (value - 100) * 0.12)}px`);
}

slider.addEventListener("input", applySize);
applySize();
