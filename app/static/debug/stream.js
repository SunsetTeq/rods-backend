const state = {
    sourceType: document.getElementById("source_type"),
    usbGroup: document.getElementById("usb_group"),
    textGroup: document.getElementById("text_group"),
    usbSource: document.getElementById("usb_source"),
    textSource: document.getElementById("text_source"),
    form: document.getElementById("camera_form"),
    errorBox: document.getElementById("error_box"),
    lastError: document.getElementById("last_error"),
    isRunning: document.getElementById("is_running"),
    sourceText: document.getElementById("source_text"),
    frameSize: document.getElementById("frame_size"),
    fps: document.getElementById("fps"),
    detectionsCount: document.getElementById("detections_count"),
    inferenceMs: document.getElementById("inference_ms"),
    visionErrorBox: document.getElementById("vision_error_box"),
    visionLastError: document.getElementById("vision_last_error"),
};

function syncSourceInputs() {
    const isUsb = state.sourceType.value === "usb";
    state.usbGroup.classList.toggle("hidden", !isUsb);
    state.textGroup.classList.toggle("hidden", isUsb);
}

function renderStatus(status) {
    state.isRunning.textContent = String(status.is_running);
    state.sourceText.textContent = `${status.source_type}:${status.source}`;
    state.frameSize.textContent = `${status.frame_width ?? "-"}x${status.frame_height ?? "-"}`;
    state.fps.textContent = String(status.actual_fps);
    state.lastError.textContent = status.last_error || "";
    state.errorBox.classList.toggle("hidden", !status.last_error);

    state.sourceType.value = status.source_type;
    if (status.source_type === "usb") {
        state.usbSource.value = status.source;
    } else {
        state.textSource.value = status.source;
    }
    syncSourceInputs();
}

function renderVisionStatus(status, detections) {
    state.detectionsCount.textContent = String(detections.detections_count ?? 0);
    state.inferenceMs.textContent = `${status.last_inference_ms ?? 0} ms`;
    state.visionLastError.textContent = status.last_error || "";
    state.visionErrorBox.classList.toggle("hidden", !status.last_error);
}

async function loadUsbSources(selectedSource) {
    const response = await fetch("/api/v1/stream/sources/usb");
    const cameras = await response.json();

    state.usbSource.innerHTML = "";

    cameras.forEach((camera) => {
        const option = document.createElement("option");
        option.value = String(camera.index);
        option.textContent =
            `${camera.label} | available=${String(camera.available).toLowerCase()} | ` +
            `${camera.width ?? "-"}x${camera.height ?? "-"}`;

        if (String(camera.index) === String(selectedSource)) {
            option.selected = true;
        }

        state.usbSource.appendChild(option);
    });
}

async function refreshStatus() {
    const [streamResponse, visionResponse, detectionsResponse] = await Promise.all([
        fetch("/api/v1/stream/status"),
        fetch("/api/v1/vision/status"),
        fetch("/api/v1/vision/detections/latest"),
    ]);

    const status = await streamResponse.json();
    const visionStatus = await visionResponse.json();
    const detections = await detectionsResponse.json();

    renderStatus(status);
    renderVisionStatus(visionStatus, detections);
}

async function initializePage() {
    const [statusResponse, visionResponse, detectionsResponse] = await Promise.all([
        fetch("/api/v1/stream/status"),
        fetch("/api/v1/vision/status"),
        fetch("/api/v1/vision/detections/latest"),
    ]);

    const status = await statusResponse.json();
    const visionStatus = await visionResponse.json();
    const detections = await detectionsResponse.json();
    await loadUsbSources(status.source);
    renderStatus(status);
    renderVisionStatus(visionStatus, detections);
}

state.sourceType.addEventListener("change", syncSourceInputs);
state.sourceType.addEventListener("change", async () => {
    if (state.sourceType.value === "usb") {
        await loadUsbSources(state.usbSource.value || "0");
    }
});

state.form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = {
        source_type: state.sourceType.value,
        source: state.sourceType.value === "usb"
            ? state.usbSource.value
            : state.textSource.value.trim(),
    };

    const response = await fetch("/api/v1/stream/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    const result = await response.json();
    if (!result.ok && result.error) {
        alert(result.error);
    }

    if (payload.source_type === "usb") {
        await loadUsbSources(payload.source);
    }
    await refreshStatus();
});

initializePage();
window.setInterval(refreshStatus, 2000);
