const API_BASE = ""; // sengaja kosong: frontend di-serve dari origin yang sama oleh FastAPI

const state = {
  mode: "image",      // "image" | "video"
  file: null,
  modelId: null,
  confThreshold: 0.5, // 0.5 - 1.0, dikontrol lewat slider (tampil sebagai 50% - 100%)
};

const el = {
  modelGrid: document.getElementById("modelGrid"),
  confSlider: document.getElementById("confSlider"),
  confValue: document.getElementById("confValue"),
  confHint: document.getElementById("confHint"),
  tabBtns: document.querySelectorAll(".tab-btn"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  dzHint: document.getElementById("dzHint"),
  fileInfo: document.getElementById("fileInfo"),
  fileName: document.getElementById("fileName"),
  clearFile: document.getElementById("clearFile"),
  runBtn: document.getElementById("runBtn"),
  progressWrap: document.getElementById("progressWrap"),
  progressStatus: document.getElementById("progressStatus"),
  progressPct: document.getElementById("progressPct"),
  progressBar: document.getElementById("progressBar"),
  progressFrames: document.getElementById("progressFrames"),
  resultSection: document.getElementById("resultSection"),
  resultImage: document.getElementById("resultImage"),
  resultVideo: document.getElementById("resultVideo"),
  statRow: document.getElementById("statRow"),
  plateList: document.getElementById("plateList"),
};

init();

async function init() {
  await loadModels();
  bindConfSlider();
  bindTabs();
  bindDropzone();
  el.clearFile.addEventListener("click", resetFile);
  el.runBtn.addEventListener("click", runDetection);
}

async function loadModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  const models = await res.json();
  state.modelId = models[0]?.id;
  renderModelCards(models);
}

function renderModelCards(models) {
  el.modelGrid.innerHTML = models
    .map(
      (m) => `
      <button type="button" class="model-card ${m.id === state.modelId ? "active" : ""}" data-model-id="${m.id}">
        <span class="mc-radio"></span>
        <p class="mc-label">${m.label}</p>
        <p class="mc-desc">${m.description}</p>
      </button>`
    )
    .join("");

  el.modelGrid.querySelectorAll(".model-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.modelId = card.dataset.modelId;
      el.modelGrid.querySelectorAll(".model-card").forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
    });
  });
}

function bindConfSlider() {
  const applyValue = (val) => {
    state.confThreshold = val / 100;
    el.confValue.textContent = `${val}%`;
    el.confSlider.style.setProperty("--fill", `${val}%`);
    el.confHint.textContent =
      val <= 60
        ? `Deteksi dengan kepercayaan di bawah ${val}% akan diabaikan. Lebih banyak deteksi, tapi lebih rawan salah.`
        : val >= 90
        ? `Deteksi dengan kepercayaan di bawah ${val}% akan diabaikan. Hasil lebih ketat & akurat, tapi bisa melewatkan objek.`
        : `Deteksi dengan kepercayaan di bawah ${val}% akan diabaikan.`;
  };

  applyValue(Number(el.confSlider.value));

  el.confSlider.addEventListener("input", (e) => {
    applyValue(Number(e.target.value));
    el.confValue.classList.add("bump");
    setTimeout(() => el.confValue.classList.remove("bump"), 120);
  });
}

function bindTabs() {
  el.tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      el.tabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      el.fileInput.accept = state.mode === "image" ? "image/*" : "video/*";
      el.dzHint.textContent = state.mode === "image" ? "JPG / PNG" : "MP4 / MOV";
      resetFile();
    });
  });
}

function bindDropzone() {
  el.dropzone.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

  ["dragover", "dragleave", "drop"].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => e.preventDefault())
  );
  el.dropzone.addEventListener("dragover", () => el.dropzone.classList.add("dragover"));
  el.dropzone.addEventListener("dragleave", () => el.dropzone.classList.remove("dragover"));
  el.dropzone.addEventListener("drop", (e) => {
    el.dropzone.classList.remove("dragover");
    setFile(e.dataTransfer.files[0]);
  });
}

function setFile(file) {
  if (!file) return;
  state.file = file;
  el.fileName.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  el.dropzone.classList.add("hidden");
  el.fileInfo.classList.remove("hidden");
  el.runBtn.disabled = false;
}

function resetFile() {
  state.file = null;
  el.fileInput.value = "";
  el.dropzone.classList.remove("hidden");
  el.fileInfo.classList.add("hidden");
  el.runBtn.disabled = true;
  el.resultSection.classList.add("hidden");
  el.progressWrap.classList.add("hidden");
}

async function runDetection() {
  el.runBtn.disabled = true;
  el.resultSection.classList.add("hidden");

  if (state.mode === "image") {
    await runImageDetection();
  } else {
    await runVideoDetection();
  }

  el.runBtn.disabled = false;
}

async function runImageDetection() {
  setProgress("processing", 30, "Menjalankan deteksi…");
  el.progressWrap.classList.remove("hidden");

  const form = new FormData();
  form.append("file", state.file);
  form.append("model_id", state.modelId);
  form.append("conf_threshold", state.confThreshold);

  try {
    const res = await fetch(`${API_BASE}/api/detect/image`, { method: "POST", body: form });
    const data = await safeJson(res);
    if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);

    setProgress("done", 100, "Selesai");
    showImageResult(data);
  } catch (err) {
    setProgress("error", 0, err.message);
  }
}

// fetch response bisa jadi bukan JSON (misal error 500 polos dari server) —
// ini mencegah error cryptic "Unexpected token..." dan nampilin pesan yang jelas.
async function safeJson(res) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text.slice(0, 200) || `Server error (${res.status})` };
  }
}

async function runVideoDetection() {
  const form = new FormData();
  form.append("file", state.file);
  form.append("model_id", state.modelId);
  form.append("conf_threshold", state.confThreshold);

  el.progressWrap.classList.remove("hidden");
  setProgress("processing", 0, "Mengunggah video…");

  const startRes = await fetch(`${API_BASE}/api/detect/video/start`, { method: "POST", body: form });
  const startData = await safeJson(startRes);
  if (!startRes.ok) {
    setProgress("error", 0, startData.detail || "Gagal memulai proses");
    return;
  }
  const { job_id } = startData;

  const poll = setInterval(async () => {
    const res = await fetch(`${API_BASE}/api/detect/video/status/${job_id}`);
    const job = await safeJson(res);
    if (!res.ok) {
      clearInterval(poll);
      setProgress("error", 0, job.detail || "Gagal mengambil status");
      return;
    }

    if (job.status === "processing" || job.status === "queued") {
      setProgress("processing", job.progress, job.stage || "Memproses…");
      const frameInfo = job.total_frames ? `Frame ${job.current_frame}/${job.total_frames}` : "";
      const fpsInfo = job.inference_fps ? ` · ${job.inference_fps} FPS` : "";
      el.progressFrames.textContent = frameInfo + fpsInfo;
    } else if (job.status === "done") {
      clearInterval(poll);
      setProgress("done", 100, "Selesai");
      showVideoResult(job);
    } else if (job.status === "error") {
      clearInterval(poll);
      setProgress("error", 0, job.error || "Terjadi kesalahan");
    }
  }, 1000);
}

function setProgress(status, pct, label) {
  el.progressStatus.textContent = label;
  el.progressPct.textContent = `${Math.round(pct)}%`;
  el.progressBar.style.width = `${pct}%`;
  el.progressBar.style.background = status === "error" ? "var(--red)" : "var(--yellow)";
}

function showImageResult(data) {
  el.resultSection.classList.remove("hidden");
  el.resultVideo.classList.add("hidden");
  el.resultImage.classList.remove("hidden");
  el.resultImage.src = data.annotated_image_url;

  renderStats(data.counts, data.inference_fps);
  renderPlates(data.vehicles);
}

function showVideoResult(job) {
  el.resultSection.classList.remove("hidden");
  el.resultImage.classList.add("hidden");
  el.resultVideo.classList.remove("hidden");
  el.resultVideo.src = job.output_video_url;

  renderStats({ ...job.counts, total: job.total_vehicles }, job.inference_fps);
  renderPlates(job.vehicles);
}

function renderStats(counts, inferenceFps) {
  const labels = { plat: "Plat", motor: "Motor", mobil: "Mobil", bus: "Bus", truck: "Truk", total: "Total" };
  const cards = Object.entries(counts)
    .filter(([key]) => key !== "plat")
    .map(([key, val]) => `
      <div class="stat-card">
        <div class="num">${val}</div>
        <div class="label">${labels[key] || key}</div>
      </div>
    `)
    .join("");

  // Kartu FPS ditaruh terakhir & tampil cuma kalau ada angkanya (video yang
  // gagal di tengah jalan sebelum PASS 1 selesai bisa aja belum punya ini).
  const fpsCard = inferenceFps
    ? `
      <div class="stat-card">
        <div class="num">${inferenceFps}</div>
        <div class="label">FPS Inferensi</div>
      </div>`
    : "";

  el.statRow.innerHTML = cards + fpsCard;
}

const JENIS_LABEL = { motor: "Motor", mobil: "Mobil", bus: "Bus", truck: "Truk" };

// vehicles: list of {id, jenis_kendaraan, plat_nomor, asal_plat, confidence}
// -- bentuk yang sama dipakai backend baik buat /api/detect/image (field
// "vehicles" di response) maupun status job video (field "vehicles" juga).
function renderPlates(vehicles) {
  const withPlate = (vehicles || []).filter((v) => v.plat_nomor);
  if (!withPlate.length) {
    el.plateList.innerHTML = `<div class="plate-empty">Belum ada plat yang berhasil dibaca.</div>`;
    return;
  }
  el.plateList.innerHTML = withPlate
    .map((v) => {
      const jenisKey = (v.jenis_kendaraan || "").toLowerCase();
      const jenisLabel = JENIS_LABEL[jenisKey] || v.jenis_kendaraan || "—";
      return `
      <div class="vehicle-row">
        <span class="vehicle-id">${v.id != null ? `#${v.id}` : "—"}</span>
        <span class="vehicle-type">
          <span class="type-dot type-${jenisKey}"></span>
          <span class="type-text">
            ${jenisLabel}
            ${v.asal_plat ? `<span class="type-region">${v.asal_plat}</span>` : ""}
          </span>
        </span>
        <span class="plate-chip">
          ${v.plat_nomor}<span class="conf">${Math.round((v.confidence || 0) * 100)}%</span>
        </span>
      </div>`;
    })
    .join("");
}