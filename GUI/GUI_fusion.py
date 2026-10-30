
"""Multimodal Lung Disease Classifier — CT + X-ray Fusion.

Inputs:
    CT: ct_v5_preprocessed.npz
    X-ray: image_XX_model_b_input.npy
           image_XX_model_c_input.npy

Standalone CT/X-ray predictions are diagnostic.
Final classification uses the frozen fusion model.

Research prototype. Not for clinical diagnosis.
"""

import html
import sys
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import tensorflow as tf
import torch
from PIL import Image


# ============================================================
# 1. CONFIGURATION — PROJECT-RELATIVE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
ENSEMBLE = ROOT / "ensemble"
SRC = ENSEMBLE / "src"

MODELS = (
    ENSEMBLE / "output" / "handover"
    / "CT_Xray_GUI_Handover" / "models"
)

CT_MODEL = MODELS / "ct" / "v61_13a378082055_best.pt"
XRAY_B = MODELS / "xray" / "model_B_v2_27sep.keras"
XRAY_C = MODELS / "xray" / "modelC_resnet50_18oct.keras"
FUSION_MODEL = (
    MODELS / "fusion" / "ct_b_xray_logistic_frozen.joblib"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.5
CT_SHAPE = (64, 192, 192)
XRAY_SHAPE = (224, 224, 3)
CT_VIEWS = 12

# Confirm these positive-class mappings against training metadata.
XRAY_B_POSITIVE = True
XRAY_C_POSITIVE = True

if not SRC.is_dir():
    raise FileNotFoundError(f"Missing source directory: {SRC}")

sys.path.insert(0, str(SRC))
from ct_v61_frozen_baseline import load_checkpoint


# ============================================================
# 2. UTILITIES
# ============================================================

def require_file(path):
    """Check model availability."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing model: {path}")
    return path


def sync_gpu():
    """Synchronize PyTorch CUDA for accurate timing."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def validate_array(array, shape, name):
    """Validate normalized preprocessed tensors."""
    array = np.asarray(array)

    if array.shape != shape:
        raise ValueError(
            f"{name}: expected {shape}, got {array.shape}"
        )

    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name}: expected floating-point data.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains invalid values.")

    if array.min() < -0.01 or array.max() > 1.01:
        raise ValueError(f"{name}: expected values within [0,1].")

    return np.ascontiguousarray(array, dtype=np.float32)


def covid_probability(output, name, positive_class=True):
    """Read a sigmoid or two-class Softmax prediction."""
    values = np.asarray(output).reshape(-1)

    if not np.isfinite(values).all():
        raise ValueError(f"{name}: invalid classifier output.")

    if values.size == 1:
        probability = float(values[0])
        if not positive_class:
            probability = 1.0 - probability

    elif values.size == 2:
        if not np.isclose(values.sum(), 1.0, atol=1e-3):
            raise ValueError(f"{name}: expected Softmax output.")
        probability = float(values[1 if positive_class else 0])

    else:
        raise ValueError(f"{name}: unsupported output shape.")

    if not 0 <= probability <= 1:
        raise ValueError(f"{name}: invalid probability.")

    return probability


# ============================================================
# 3. KERAS FEATURE + CLASSIFICATION ADAPTER
# ============================================================

def build_xray_encoder(model, name):
    """Expose 128D features and final classification output.

    Reuses the original Sequential layers and trained weights.
    """
    inputs = tf.keras.Input(
        shape=XRAY_SHAPE,
        name=f"{name}_input",
    )

    x = inputs
    features = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        x = layer(x, training=False)

        if layer.name == "dense":
            features = x

    if features is None:
        raise ValueError(f"{name}: missing 128D dense layer.")

    encoder = tf.keras.Model(
        inputs=inputs,
        outputs=[features, x],
        name=f"{name}_encoder",
    )
    encoder.trainable = False

    if encoder.output_shape[0][-1] != 128:
        raise ValueError(f"{name}: expected 128D features.")

    return encoder


# ============================================================
# 4. LOAD FROZEN MODELS
# ============================================================

@lru_cache(maxsize=1)
def load_models():
    """Load checkpoints once and cache them."""
    ct, config = load_checkpoint(
        require_file(CT_MODEL), DEVICE
    )

    if (
        config["arch"] != "densenet121"
        or config["pooling"] != "mean"
        or config["views"] != CT_VIEWS
    ):
        raise ValueError("Unexpected CT encoder configuration.")

    full_b = tf.keras.models.load_model(
        str(require_file(XRAY_B)), compile=False
    )
    full_c = tf.keras.models.load_model(
        str(require_file(XRAY_C)), compile=False
    )

    b = build_xray_encoder(full_b, "model_b")
    c = build_xray_encoder(full_c, "model_c")

    # Load only trusted teammate artifacts.
    fusion = joblib.load(require_file(FUSION_MODEL))

    if list(fusion.named_steps) != ["scale", "clf"]:
        raise ValueError("Unexpected fusion pipeline.")

    if fusion.named_steps["scale"].n_features_in_ != 1280:
        raise ValueError("Fusion requires 1280D input.")

    if list(fusion.classes_) != [0, 1]:
        raise ValueError("Unexpected fusion class mapping.")

    return {"ct": ct, "b": b, "c": c, "fusion": fusion}


# ============================================================
# 5. PREPROCESSED INPUTS
# ============================================================

def load_ct(path):
    """Load CT V5 volume."""
    if not path:
        raise ValueError("CT input required.")

    with np.load(path, allow_pickle=False) as data:
        if "volume" in data:
            volume = data["volume"]
        elif "tensor" in data:
            volume = data["tensor"]
        else:
            raise ValueError(
                f"Unknown CT archive keys: {data.files}"
            )

    return validate_array(volume, CT_SHAPE, "CT")


def load_xray(path):
    """Load preprocessed X-ray tensor."""
    return validate_array(
        np.load(path, allow_pickle=False),
        XRAY_SHAPE,
        Path(path).name,
    )


def pair_xrays(b_files, c_files):
    """Match X-ray Model B/C files by image identifier."""
    if not b_files or not c_files:
        raise ValueError("Both X-ray inputs are required.")

    def index(files, model):
        result = {}
        suffix = f"_model_{model}_input.npy"

        for path in files:
            name = Path(path).name.lower()

            if not name.endswith(suffix):
                raise ValueError(f"Invalid filename: {name}")

            key = name[:-len(suffix)]

            if key in result:
                raise ValueError(f"Duplicate image: {key}")

            result[key] = path

        return result

    b = index(b_files, "b")
    c = index(c_files, "c")

    if set(b) != set(c):
        raise ValueError("Model B/C image identifiers do not match.")

    return [(key, b[key], c[key]) for key in sorted(b)]


# ============================================================
# 6. IMAGE PREVIEWS
# ============================================================

def preview_ct(path):
    """Display the middle CT slice."""
    if not path:
        return None

    volume = load_ct(path)
    image = volume[volume.shape[0] // 2]

    return Image.fromarray(
        (np.clip(image, 0, 1) * 255).astype(np.uint8)
    )


def preview_xray(files):
    """Display first uploaded X-ray."""
    if not files:
        return None

    image = load_xray(files[0])

    return Image.fromarray(
        (np.clip(image, 0, 1) * 255).astype(np.uint8)
    )


# ============================================================
# 7. CT FEATURE EXTRACTION
# ============================================================

def extract_ct(path, model):
    """Return 1024D CT features and COVID probability."""
    volume = torch.from_numpy(load_ct(path))

    centers = torch.linspace(
        3, 60, CT_VIEWS
    ).round().long()

    views = torch.stack([
        volume[c - 1:c + 2] for c in centers
    ]).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        features, logits = model.features_and_logits(views)
        probability = float(
            logits.float().softmax(-1)[0, 1].cpu()
        )

    vector = features[0].float().cpu().numpy()

    if vector.shape != (1024,) or not np.isfinite(vector).all():
        raise ValueError("Invalid CT features.")

    return vector.astype(np.float32), probability


# ============================================================
# 8. X-RAY FEATURE EXTRACTION
# ============================================================

def extract_xray(pairs, model_b, model_c):
    """Return 256D features and standalone X-ray probability."""
    vectors = []
    probabilities = []

    for key, b_path, c_path in pairs:
        b_input = load_xray(b_path)
        c_input = load_xray(c_path)

        b_features, b_output = model_b(
            b_input[None, ...], training=False
        )
        c_features, c_output = model_c(
            c_input[None, ...], training=False
        )

        b_vec = np.asarray(b_features).reshape(-1)
        c_vec = np.asarray(c_features).reshape(-1)

        if b_vec.shape != (128,) or c_vec.shape != (128,):
            raise ValueError(f"{key}: invalid feature dimensions.")

        vector = np.concatenate([b_vec, c_vec]).astype(np.float32)

        if not np.isfinite(vector).all():
            raise ValueError(f"{key}: invalid feature values.")

        vectors.append(vector)

        # Equal-weight soft voting for diagnostic probability.
        p_b = covid_probability(
            b_output, "Model B", XRAY_B_POSITIVE
        )
        p_c = covid_probability(
            c_output, "Model C", XRAY_C_POSITIVE
        )
        probabilities.append((p_b + p_c) / 2)

    features = np.mean(
        np.stack(vectors), axis=0
    ).astype(np.float32)

    if features.shape != (256,):
        raise ValueError("Invalid X-ray features.")

    return features, float(np.mean(probabilities))


# ============================================================
# 9. FROZEN FUSION INFERENCE + TIMING
# ============================================================

def predict_fusion(ct_path, b_files, c_files):
    """Calculate standalone and final fusion predictions."""
    start_total = time.perf_counter()

    pairs = pair_xrays(b_files, c_files)

    start = time.perf_counter()
    models = load_models()
    sync_gpu()
    load_time = time.perf_counter() - start

    # CT encoder.
    sync_gpu()
    start = time.perf_counter()
    ct_features, ct_positive = extract_ct(
        ct_path, models["ct"]
    )
    sync_gpu()
    ct_time = time.perf_counter() - start

    # X-ray encoders.
    sync_gpu()
    start = time.perf_counter()
    xray_features, xray_positive = extract_xray(
        pairs, models["b"], models["c"]
    )
    sync_gpu()
    xray_time = time.perf_counter() - start

    # Frozen CT + X-ray fusion classifier.
    start = time.perf_counter()

    fusion_input = np.concatenate([
        ct_features, xray_features
    ]).astype(np.float64)[None, :]

    if fusion_input.shape != (1, 1280):
        raise ValueError("Invalid fusion input dimensions.")

    probabilities = np.asarray(
        models["fusion"].predict_proba(fusion_input)[0]
    )

    if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid fusion probabilities.")

    p_negative, p_positive = map(float, probabilities)
    fusion_time = time.perf_counter() - start

    positive = p_positive >= THRESHOLD

    return {
        "label": "COVID POSITIVE" if positive else "COVID NEGATIVE",
        "positive": positive,
        "confidence": p_positive if positive else p_negative,
        "ct_positive": ct_positive,
        "xray_positive": xray_positive,
        "xray_count": len(pairs),
        "ct_time": ct_time,
        "xray_time": xray_time,
        "fusion_time": fusion_time,
        "load_time": load_time,
        "total_time": time.perf_counter() - start_total,
    }


# ============================================================
# 10. CLASSIFICATION RESULTS
# ============================================================

def classify(ct_path, b_files, c_files):
    """Display fusion cards and encoder diagnostics."""
    try:
        r = predict_fusion(ct_path, b_files, c_files)
        status = "positive" if r["positive"] else "negative"

        return f"""
        <div class="results-grid">
            <div class="result-card classification-{status}">
                <div class="result-heading">Classification</div>
                <div class="result-value">{r["label"]}</div>
            </div>
            <div class="result-card">
                <div class="result-heading">Confidence</div>
                <div class="result-value">{r["confidence"]:.0%}</div>
            </div>
        </div>

        <div class="model-information">
            <h3>Model Information</h3>
            <table>
                <thead>
                    <tr>
                        <th>Information</th>
                        <th>CT Encoder</th>
                        <th>X-ray Ensemble</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Architecture</td>
                        <td>DenseNet121 V6.1 Mean</td>
                        <td>Model B + Model C</td>
                    </tr>
                    <tr>
                        <td>Images processed</td>
                        <td>12 three-slice views</td>
                        <td>{r["xray_count"]} X-ray(s)</td>
                    </tr>
                    <tr>
                        <td>Features extracted</td>
                        <td>1024D</td>
                        <td>256D (128D + 128D)</td>
                    </tr>
                    <tr>
                        <td>COVID Positive probability</td>
                        <td>{r["ct_positive"]:.0%}</td>
                        <td>{r["xray_positive"]:.0%}</td>
                    </tr>
                    <tr>
                        <td>COVID Negative probability</td>
                        <td>{1-r["ct_positive"]:.0%}</td>
                        <td>{1-r["xray_positive"]:.0%}</td>
                    </tr>
                    <tr>
                        <td>Processing time</td>
                        <td>{r["ct_time"]:.3f} s</td>
                        <td>{r["xray_time"]:.3f} s</td>
                    </tr>
                    <tr>
                        <td>Fusion classifier time</td>
                        <td colspan="2">{r["fusion_time"]:.3f} s</td>
                    </tr>
                    <tr>
                        <td>Model loading time</td>
                        <td colspan="2">{r["load_time"]:.3f} s</td>
                    </tr>
                    <tr>
                        <td>Total processing time</td>
                        <td colspan="2">{r["total_time"]:.3f} s</td>
                    </tr>
                </tbody>
            </table>
            <p class="model-note">
                Standalone X-ray probabilities use equal-weight
                soft voting. Final classification uses the frozen
                CT + X-ray feature fusion model.
            </p>
        </div>
        """

    except Exception as exc:
        message = html.escape(
            f"{type(exc).__name__}: {exc}"
        )
        return f"<h3>Inference failed</h3><p>{message}</p>"


# ============================================================
# 11. PROFESSIONAL HOSPITAL THEME
# ============================================================

# Gradio handles native upload, preview, filename and icon styling.
# Custom CSS is limited to hospital branding and result presentation.

LIGHT_THEME = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="teal",
    neutral_hue="slate",
)

CSS = """
/* ============================================================
   PAGE LAYOUT
   ============================================================ */

.gradio-container {
    max-width: none !important;
    width: 100% !important;
    padding: 12px 24px !important;
}


/* ============================================================
   HOSPITAL HEADER
   ============================================================ */

.hospital-header {
    background: #075985;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 10px;
}

.hospital-header,
.hospital-header * {
    color: #FFFFFF !important;
}

.hospital-header h1 {
    font-size: 25px !important; /* Main title */
    margin: 0;
}

.hospital-header p {
    font-size: 14px !important; /* Subtitle */
    margin: 3px 0 0;
}


/* ============================================================
   INPUT PANELS
   ============================================================ */

.clinical-panel {
    border: 1px solid #CBD5E1;
    border-top: 3px solid #0D9488;
    border-radius: 9px;
    padding: 12px;
}

/* Native Gradio File and Image components remain unmodified. */


/* ============================================================
   CLASSIFICATION BUTTON
   ============================================================ */

#run-button {
    background: #D7F0F7 !important;
    color: #075985 !important;
    border: 1px solid #9BD3E3 !important;
    border-radius: 8px;
    font-size: 16px !important; /* Button font size */
    font-weight: 700;
    min-height: 40px;
}

#run-button:hover {
    background: #BFE5F0 !important;
}


/* ============================================================
   RESULTS PANEL
   ============================================================ */

.clinical-output {
    border: 1px solid #CBD5E1;
    border-radius: 9px;
    padding: 12px;
}

/* Two equal-width result cards */
.results-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    width: 100%;
}

.result-card {
    min-width: 0;
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-top: 3px solid #0D9488;
    border-radius: 9px;
    padding: 14px 8px;
    text-align: center;
}

/* COVID Positive */
.result-card.classification-positive {
    background: #FFF1F2;
    border-color: #FDA4AF;
    border-top-color: #DC2626;
}

.result-card.classification-positive .result-value {
    color: #B91C1C;
}

/* COVID Negative */
.result-card.classification-negative {
    background: #F0FDF4;
    border-color: #86EFAC;
    border-top-color: #16A34A;
}

.result-card.classification-negative .result-value {
    color: #15803D;
}

/* RESULT CARD FONT SIZES */
.result-heading {
    font-size: 15px; /* Card heading */
    font-weight: 600;
    color: #64748B;
    margin-bottom: 6px;
}

.result-value {
    font-size: 28px; /* Classification and confidence */
    font-weight: 700;
    color: #075985;
    overflow-wrap: anywhere;
}


/* ============================================================
   MODEL INFORMATION TABLE
   ============================================================ */

.model-information {
    margin-top: 12px;
    width: 100%;
}

.model-information h3 {
    font-size: 18px; /* Table title */
    color: #075985;
    margin: 0 0 8px;
}

.model-information table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 15px; /* Table body */
    background: #FFFFFF;
    color: #172033;
}

.model-information th {
    background: #EDF5FA;
    color: #075985;
    font-size: 15px; /* Column headings */
    font-weight: 700;
    text-align: left;
}

.model-information td {
    background: #FFFFFF;
    color: #172033;
}

.model-information th,
.model-information td {
    padding: 9px 10px;
    border: 1px solid #CBD5E1;
    overflow-wrap: anywhere;
}

.model-information th:first-child {
    width: 34%;
}

.model-information th:nth-child(2),
.model-information th:nth-child(3) {
    width: 33%;
}


/* ============================================================
   FOOTER
   ============================================================ */

.model-note,
.clinical-footer {
    font-size: 12px;
    color: #64748B;
}


/* ============================================================
   RESPONSIVE LAYOUT
   ============================================================ */

@media (max-width: 900px) {
    .gradio-container {
        padding: 8px !important;
    }

    .result-value {
        font-size: 22px;
    }

    .model-information table,
    .model-information th {
        font-size: 13px;
    }

    .model-information th,
    .model-information td {
        padding: 6px;
    }
}
"""


# ============================================================
# 12. GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    theme=LIGHT_THEME,
    css=CSS,
    title="Multimodal Lung Disease Classifier",
) as demo:

    gr.Markdown(
        "# Multimodal Lung Disease Classifier\n"
        "CT + X-ray | Frozen Fusion Model",
        elem_classes="hospital-header",
    )

    with gr.Row(equal_height=True):

        with gr.Column(elem_classes="clinical-panel"):
            xray_b_input = gr.File(
                label="X-ray Model B input (.npy)",
                file_types=[".npy"],
                type="filepath",
                file_count="multiple",
            )

            xray_b_preview = gr.Image(
                label="X-ray Model B Preview",
                interactive=False,
                height=240,
            )

        with gr.Column(elem_classes="clinical-panel"):
            xray_c_input = gr.File(
                label="X-ray Model C input (.npy)",
                file_types=[".npy"],
                type="filepath",
                file_count="multiple",
            )

            xray_c_preview = gr.Image(
                label="X-ray Model C Preview",
                interactive=False,
                height=240,
            )

        with gr.Column(elem_classes="clinical-panel"):
            ct_input = gr.File(
                label="CT input (.npz)",
                file_types=[".npz"],
                type="filepath",
                file_count="single",
            )

            ct_preview = gr.Image(
                label="CT Preview (Middle Slice)",
                interactive=False,
                height=240,
            )

    # Preview events do not alter inference tensors.
    ct_input.change(
        preview_ct, ct_input, ct_preview
    )

    xray_b_input.change(
        preview_xray, xray_b_input, xray_b_preview
    )

    xray_c_input.change(
        preview_xray, xray_c_input, xray_c_preview
    )

    run_button = gr.Button(
        "Run Fusion Classification",
        elem_id="run-button",
    )

    with gr.Column(elem_classes="clinical-output"):
        output = gr.HTML(
            "Upload the preprocessed CT volume and "
            "matching Model B/C X-ray tensors."
        )

    run_button.click(
        classify,
        inputs=[ct_input, xray_b_input, xray_c_input],
        outputs=output,
    )

    gr.Markdown(
        "Research prototype. Inputs must belong to the "
        "same patient. Not for clinical diagnosis.",
        elem_classes="clinical-footer",
    )



# ============================================================
# 13. START APPLICATION — LIGHT MODE DEFAULT
# ============================================================

if __name__ == "__main__":
    import webbrowser

    # Launch Gradio without opening its default browser URL.
    demo.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
        prevent_thread_lock=True,
    )

    # Open the confirmed native Gradio Light appearance.
    webbrowser.open(
        "http://127.0.0.1:7860/?__theme=light"
    )

