/**
 * Laura Image Studio - Control Center Web Extension
 *
 * Intercepts the LauraControlCenter node to render a rich status panel
 * showing VRAM info, model status badges, and download buttons.
 * Uses safe DOM methods exclusively (no innerHTML with dynamic data).
 *
 * Pattern: ComfyUI web extension using app.registerExtension()
 */

import { app } from "../../scripts/app.js";

// ---------------------------------------------------------------------------
// HuggingFace repo mapping (model_key -> org/repo)
// ---------------------------------------------------------------------------

const REPO_MAP = {
    "z_image_turbo": "Tongyi-MAI/Z-Image-Turbo",
    "z_image_base": "Tongyi-MAI/Z-Image-Base",
    "flux1_dev": "black-forest-labs/FLUX.1-dev",
    "flux1_schnell": "black-forest-labs/FLUX.1-schnell",
    "flux2_dev": "black-forest-labs/FLUX.2-dev",
    "sd35_medium": "stabilityai/stable-diffusion-3.5-medium",
    "wan22_14b": "Wan-AI/Wan2.2-T2V-14B",
    "hunyuan_video": "tencent/HunyuanVideo",
    "cogvideox_5b": "THUDM/CogVideoX-5b",
    "helios_distilled": "BestWishYsh/Helios-Distilled",
    "helios_base": "BestWishYsh/Helios-Base",
    "ltx_2_3": "Lightricks/LTX-Video",
    "seedvr2": "ByteDance-Seed/SeedVR2-7B",
    "qwen_image_2512": "Qwen/Qwen-Image-2512",
    "glm_image": "THUDM/GLM-Image",
    "firered_edit_1_1": "FireRedTeam/FireRed-Image-Edit",
};

// ---------------------------------------------------------------------------
// Styles (injected once)
// ---------------------------------------------------------------------------

let stylesInjected = false;

function injectStyles() {
    if (stylesInjected) return;
    stylesInjected = true;

    const style = document.createElement("style");
    style.textContent = [
        "/* Laura Control Center - Status Panel */",
        ".laura-cc-panel {",
        "    font-family: 'Segoe UI', Arial, sans-serif;",
        "    font-size: 11px;",
        "    line-height: 1.6;",
        "    color: #e0e0e0;",
        "    background: #1a1a2e;",
        "    border: 1px solid #333;",
        "    border-radius: 6px;",
        "    padding: 10px;",
        "    max-height: 420px;",
        "    overflow-y: auto;",
        "}",
        ".laura-cc-section {",
        "    margin-bottom: 8px;",
        "}",
        ".laura-cc-label {",
        "    color: #888;",
        "    font-size: 10px;",
        "    text-transform: uppercase;",
        "    letter-spacing: 0.5px;",
        "    margin-bottom: 2px;",
        "}",
        ".laura-cc-vram {",
        "    color: #5ba0f7;",
        "    font-weight: bold;",
        "    font-size: 13px;",
        "}",
        ".laura-cc-quant {",
        "    color: #b48efa;",
        "    font-weight: bold;",
        "}",
        ".laura-cc-model-val {",
        "    color: #d0d0d0;",
        "}",
        ".laura-cc-sep {",
        "    border: none;",
        "    border-top: 1px solid #333;",
        "    margin: 8px 0;",
        "}",
        ".laura-cc-file {",
        "    font-family: 'Courier New', monospace;",
        "    font-size: 10px;",
        "    padding: 2px 0;",
        "}",
        ".laura-cc-installed {",
        "    color: #4caf50;",
        "}",
        ".laura-cc-missing {",
        "    color: #f44336;",
        "}",
        ".laura-cc-btn-row {",
        "    display: flex;",
        "    gap: 6px;",
        "    margin-top: 10px;",
        "}",
        ".laura-cc-btn {",
        "    padding: 5px 12px;",
        "    border: none;",
        "    border-radius: 4px;",
        "    cursor: pointer;",
        "    font-size: 11px;",
        "    font-weight: 600;",
        "    transition: opacity 0.15s ease;",
        "}",
        ".laura-cc-btn:hover { opacity: 0.85; }",
        ".laura-cc-btn-hf {",
        "    background: #ffd700;",
        "    color: #1a1a2e;",
        "}",
        ".laura-cc-btn-dl {",
        "    background: #4caf50;",
        "    color: #fff;",
        "}",
        ".laura-cc-btn:disabled {",
        "    opacity: 0.5;",
        "    cursor: not-allowed;",
        "}",
        ".laura-cc-msg {",
        "    font-size: 10px;",
        "    margin-top: 4px;",
        "    min-height: 14px;",
        "}",
    ].join("\n");
    document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// Status report parser
// ---------------------------------------------------------------------------

function parseStatusReport(text) {
    const result = {
        gpu: "",
        tier: "",
        quantization: "",
        imageModel: "",
        videoModel: "",
        faceModel: "",
        files: [],
    };
    if (!text) return result;

    const lines = text.split("\n");
    for (const raw of lines) {
        const line = raw.trim();
        if (!line) continue;

        // GPU: 12.0 GB VRAM | Tier: high
        if (line.startsWith("GPU:")) {
            const tierMatch = line.match(/Tier:\s*(\S+)/);
            result.tier = tierMatch ? tierMatch[1] : "";
            result.gpu = line;
        } else if (line.startsWith("Quantization:")) {
            result.quantization = line.replace("Quantization:", "").trim();
        } else if (line.startsWith("Image Model:")) {
            result.imageModel = line.replace("Image Model:", "").trim();
        } else if (line.startsWith("Video Model:")) {
            result.videoModel = line.replace("Video Model:", "").trim();
        } else if (line.startsWith("Face Model:")) {
            result.faceModel = line.replace("Face Model:", "").trim();
        } else if (line.startsWith("[INSTALLED]") || line.startsWith("[MISSING]")) {
            const installed = line.startsWith("[INSTALLED]");
            const filename = line
                .replace("[INSTALLED]", "")
                .replace("[MISSING]", "")
                .trim();
            result.files.push({ filename, installed });
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Download helper
// ---------------------------------------------------------------------------

async function lauraDownload(modelKey, msgEl) {
    msgEl.textContent = "Starting download...";
    msgEl.style.color = "#ffd700";
    try {
        const resp = await fetch("/laura/download", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_key: modelKey }),
        });
        const data = await resp.json();
        if (data.job_id) {
            msgEl.textContent = "Download started (job: " + data.job_id + ")";
            msgEl.style.color = "#4caf50";
        } else {
            msgEl.textContent = "Error: " + (data.error || "unknown error");
            msgEl.style.color = "#f44336";
        }
    } catch (e) {
        console.error("Laura download error:", e);
        msgEl.textContent = "Network error - see console";
        msgEl.style.color = "#f44336";
    }
}

// ---------------------------------------------------------------------------
// DOM builder — constructs the status panel using safe methods only
// ---------------------------------------------------------------------------

function buildPanel(status, config) {
    const panel = document.createElement("div");
    panel.className = "laura-cc-panel";

    // --- VRAM / Tier ---
    const vramSection = document.createElement("div");
    vramSection.className = "laura-cc-section";
    const vramLabel = document.createElement("div");
    vramLabel.className = "laura-cc-label";
    vramLabel.textContent = "System";
    const vramVal = document.createElement("div");
    vramVal.className = "laura-cc-vram";
    vramVal.textContent = status.gpu || "No GPU info";
    vramSection.appendChild(vramLabel);
    vramSection.appendChild(vramVal);
    panel.appendChild(vramSection);

    // --- Quantization ---
    if (status.quantization) {
        const quantSection = document.createElement("div");
        quantSection.className = "laura-cc-section";
        const quantLabel = document.createElement("div");
        quantLabel.className = "laura-cc-label";
        quantLabel.textContent = "Quantization";
        const quantVal = document.createElement("div");
        quantVal.className = "laura-cc-quant";
        quantVal.textContent = status.quantization;
        quantSection.appendChild(quantLabel);
        quantSection.appendChild(quantVal);
        panel.appendChild(quantSection);
    }

    // --- Model selections ---
    const models = [
        { label: "Image Model", value: status.imageModel },
        { label: "Video Model", value: status.videoModel },
        { label: "Face Model", value: status.faceModel },
    ];
    for (const m of models) {
        if (!m.value) continue;
        const sec = document.createElement("div");
        sec.className = "laura-cc-section";
        const lbl = document.createElement("div");
        lbl.className = "laura-cc-label";
        lbl.textContent = m.label;
        const val = document.createElement("div");
        val.className = "laura-cc-model-val";
        val.textContent = m.value;
        sec.appendChild(lbl);
        sec.appendChild(val);
        panel.appendChild(sec);
    }

    // --- Separator ---
    const sep = document.createElement("hr");
    sep.className = "laura-cc-sep";
    panel.appendChild(sep);

    // --- File status list ---
    if (status.files.length > 0) {
        const filesLabel = document.createElement("div");
        filesLabel.className = "laura-cc-label";
        filesLabel.textContent = "File Status";
        panel.appendChild(filesLabel);

        for (const f of status.files) {
            const row = document.createElement("div");
            row.className = "laura-cc-file";
            const badge = document.createElement("span");
            if (f.installed) {
                badge.className = "laura-cc-installed";
                badge.textContent = "[INSTALLED] ";
            } else {
                badge.className = "laura-cc-missing";
                badge.textContent = "[MISSING]   ";
            }
            const name = document.createElement("span");
            name.textContent = f.filename;
            row.appendChild(badge);
            row.appendChild(name);
            panel.appendChild(row);
        }
    }

    // --- Action buttons ---
    const modelKey = (config && config.image_model) || (config && config.video_model) || "";
    const repoSlug = REPO_MAP[modelKey] || "";

    const btnRow = document.createElement("div");
    btnRow.className = "laura-cc-btn-row";

    // HuggingFace link
    if (repoSlug) {
        const hfBtn = document.createElement("a");
        hfBtn.className = "laura-cc-btn laura-cc-btn-hf";
        hfBtn.href = "https://huggingface.co/" + repoSlug;
        hfBtn.target = "_blank";
        hfBtn.rel = "noopener noreferrer";
        hfBtn.textContent = "View on HuggingFace";
        hfBtn.style.textDecoration = "none";
        btnRow.appendChild(hfBtn);
    }

    // Download button
    const dlBtn = document.createElement("button");
    dlBtn.className = "laura-cc-btn laura-cc-btn-dl";
    dlBtn.textContent = "Download Missing";
    const hasMissing = status.files.some(function (f) { return !f.installed; });
    if (!hasMissing || !modelKey) {
        dlBtn.disabled = true;
        dlBtn.textContent = hasMissing ? "Download" : "All Installed";
    }
    btnRow.appendChild(dlBtn);
    panel.appendChild(btnRow);

    // Status message area
    const msgEl = document.createElement("div");
    msgEl.className = "laura-cc-msg";
    panel.appendChild(msgEl);

    // Download click handler
    dlBtn.addEventListener("click", function () {
        if (dlBtn.disabled) return;
        dlBtn.disabled = true;
        lauraDownload(modelKey, msgEl).then(function () {
            dlBtn.disabled = false;
        });
    });

    return panel;
}

// ---------------------------------------------------------------------------
// Extension Registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "laura.studio.control_center",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (!nodeData || nodeData.name !== "LauraControlCenter") {
            return;
        }

        injectStyles();

        const origOnExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function (output) {
            if (origOnExecuted) {
                origOnExecuted.call(this, output);
            }

            // Extract outputs
            const statusText =
                (output && output.status_report && output.status_report[0]) || "";
            const configText =
                (output && output.config_json && output.config_json[0]) || "";

            // Parse
            const status = parseStatusReport(statusText);
            let config = {};
            try {
                config = configText ? JSON.parse(configText) : {};
            } catch (e) {
                console.warn("Laura CC: could not parse config_json", e);
            }

            // Build the panel
            const panel = buildPanel(status, config);

            // Find or create widget container
            let wrapper = this._lauraCCWrapper;
            if (!wrapper) {
                wrapper = document.createElement("div");
                wrapper.style.width = "100%";
                this._lauraCCWrapper = wrapper;

                const widget = this.addDOMWidget(
                    "laura_cc_display",
                    "customtext",
                    wrapper,
                    {
                        serialize: false,
                        hideOnZoom: false,
                    }
                );
                widget.computeSize = function () {
                    return [360, 340];
                };
            }

            // Replace contents
            while (wrapper.firstChild) {
                wrapper.removeChild(wrapper.firstChild);
            }
            wrapper.appendChild(panel);

            // Resize node to fit
            this.setSize([
                Math.max(this.size[0], 380),
                Math.max(this.size[1], 420),
            ]);
        };
    },
});
