
"""Multimodal Lung Disease Classifier — Frozen CT + X-ray Fusion.

Inputs:
    CT:    ct_v5_preprocessed.npz
    X-ray: image_XX_model_b_input.npy
           image_XX_model_c_input.npy

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
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
ENSEMBLE_DIR = ROOT_DIR / "ensemble"
SRC_DIR = ENSEMBLE_DIR / "src"

HANDOVER_DIR = (
    ENSEMBLE_DIR / "output" / "handover"
    / "CT_Xray_GUI_Handover"
)

MODEL_DIR = HANDOVER_DIR / "models"

MODEL_PATH_CT = (
    MODEL_DIR / "ct" / "v61_13a378082055_best.pt"
)
MODEL_PATH_XRAY_B = (
    MODEL_DIR / "xray" / "model_B_v2_27sep.keras"
)
MODEL_PATH_XRAY_C = (
    MODEL_DIR / "xray" / "modelC_resnet50_18oct.keras"
)
MODEL_PATH_FUSION = (
    MODEL_DIR / "fusion" / "ct_b_xray_logistic_frozen.joblib"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.5
CT_SHAPE = (64, 192, 192)
XRAY_SHAPE = (224, 224, 3)
CT_VIEWS = 12

if not SRC_DIR.is_dir():
    raise FileNotFoundError(
        f"Fusion source directory missing: {SRC_DIR}"
    )

sys.path.insert(0, str(SRC_DIR))
from ct_v61_frozen_baseline import load_checkpoint


# ============================================================
# 2. UTILITIES
# ============================================================

def require_file(path):
    """Verify that a required model file exists."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required model file missing: {path}"
        )
    return path


def synchronize_gpu():
    """Ensure CUDA operations finish before timing."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def validate_array(array, shape, name):
    """Check shape, numeric type and normalized range."""
    array = np.asarray(array)

    if array.shape != shape:
        raise ValueError(
            f"{name}: expected {shape}, got {array.shape}"
        )

    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(
            f"{name}: expected floating-point data."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name}: contains NaN or infinite values."
        )

    if array.min() < -0.01 or array.max() > 1.01:
        raise ValueError(
            f"{name}: expected normalized range [0,1]."
        )

    return np.ascontiguousarray(
        array.astype(np.float32)
    )


# ============================================================
# 3. LOAD FROZEN MODELS
# ============================================================

@lru_cache(maxsize=1)
def load_models():
    """Load all models once and reuse them."""
    ct_path = require_file(MODEL_PATH_CT)
    b_path = require_file(MODEL_PATH_XRAY_B)
    c_path = require_file(MODEL_PATH_XRAY_C)
    fusion_path = require_file(MODEL_PATH_FUSION)

    ct_model, config = load_checkpoint(
        ct_path, DEVICE
    )

    if (
        config["arch"] != "densenet121"
        or config["pooling"] != "mean"
        or config["views"] != CT_VIEWS
    ):
        raise ValueError(
            "Unexpected CT encoder configuration."
        )

    full_b = tf.keras.models.load_model(
        str(b_path), compile=False
    )
    full_c = tf.keras.models.load_model(
        str(c_path), compile=False
    )

    for name, model in [
        ("Model B", full_b),
        ("Model C", full_c),
    ]:
        if tuple(model.input_shape[1:]) != XRAY_SHAPE:
            raise ValueError(
                f"{name}: unexpected input shape."
            )

    encoder_b = tf.keras.Model(
        full_b.inputs,
        full_b.get_layer("dense").output,
    )
    encoder_c = tf.keras.Model(
        full_c.inputs,
        full_c.get_layer("dense").output,
    )

    encoder_b.trainable = False
    encoder_c.trainable = False

    if encoder_b.output_shape[-1] != 128:
        raise ValueError("Model B must output 128D.")

    if encoder_c.output_shape[-1] != 128:
        raise ValueError("Model C must output 128D.")

    # Load only trusted teammate checkpoints.
    fusion = joblib.load(fusion_path)

    if not hasattr(fusion, "named_steps"):
        raise ValueError("Invalid fusion pipeline.")

    if list(fusion.named_steps) != ["scale", "clf"]:
        raise ValueError("Unexpected fusion steps.")

    if fusion.named_steps["scale"].n_features_in_ != 1280:
        raise ValueError("Fusion must accept 1280D.")

    if list(fusion.classes_) != [0, 1]:
        raise ValueError("Unexpected fusion class order.")

    return {
        "ct": ct_model,
        "b": encoder_b,
        "c": encoder_c,
        "fusion": fusion,
    }


# ============================================================
# 4. PREPROCESSED INPUT LOADING
# ============================================================

def load_ct_tensor(path):
    """Load preprocessed V5 CT volume."""
    if not path:
        raise ValueError("CT input is required.")

    if Path(path).suffix.lower() != ".npz":
        raise ValueError("CT must be a .npz file.")

    with np.load(path, allow_pickle=False) as data:
        if "volume" in data.files:
            volume = data["volume"]
        elif "tensor" in data.files:
            volume = data["tensor"]
        else:
            raise ValueError(
                f"Unknown CT archive keys: {data.files}"
            )

    return validate_array(
        volume, CT_SHAPE, "CT"
    )


def load_xray_tensor(path, name):
    """Load preprocessed X-ray encoder input."""
    if Path(path).suffix.lower() != ".npy":
        raise ValueError(
            f"{name}: expected .npy file."
        )

    array = np.load(path, allow_pickle=False)

    return validate_array(
        array, XRAY_SHAPE, name
    )


def image_id(path, model):
    """Identify corresponding Model B/C X-ray tensors."""
    name = Path(path).name.lower()
    suffix = f"_model_{model}_input.npy"

    if not name.endswith(suffix):
        raise ValueError(
            f"Invalid X-ray filename: {name}"
        )

    return name[:-len(suffix)]


def pair_xray_files(b_files, c_files):
    """Match B/C tensors for each original X-ray."""
    if not b_files or not c_files:
        raise ValueError(
            "Both X-ray Model B and C inputs are required."
        )

    def index_files(files, model):
        result = {}

        for path in files:
            key = image_id(path, model)

            if key in result:
                raise ValueError(
                    f"Duplicate X-ray identifier: {key}"
                )

            result[key] = path

        return result

    b = index_files(b_files, "b")
    c = index_files(c_files, "c")

    if set(b) != set(c):
        raise ValueError(
            "Model B and C X-ray filenames do not match."
        )

    return [
        (key, b[key], c[key])
        for key in sorted(b)
    ]


# ============================================================
# 4A. IMAGE PREVIEW
# ============================================================

def preview_ct(path):
    """Display the middle axial slice of a preprocessed CT."""
    if not path:
        return None

    volume = load_ct_tensor(path)  # (64, 192, 192)
    middle_slice = volume[volume.shape[0] // 2]

    image = (np.clip(middle_slice, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(image)


def preview_xray(files):
    """Display the first preprocessed X-ray image."""
    if not files:
        return None

    image = load_xray_tensor(files[0], "X-ray preview")
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(image)


# ============================================================
# 5. CT FEATURE EXTRACTION
# ============================================================

def extract_ct_features(path, model):
    """Extract 1024D CT features and CT probability."""
    volume = load_ct_tensor(path)
    x = torch.from_numpy(volume)

    centers = torch.linspace(
        3, 60, CT_VIEWS
    ).round().long()

    views = torch.stack([
        x[c - 1:c + 2] for c in centers
    ])

    views = views.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        features, logits = model.features_and_logits(views)

        probability = float(
            logits.float().softmax(-1)[0, 1].cpu()
        )

    vector = (
        features[0]
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if vector.shape != (1024,):
        raise ValueError("Invalid CT feature dimensions.")

    if not np.isfinite(vector).all():
        raise ValueError("Invalid CT feature values.")

    return vector, probability


# ============================================================
# 6. X-RAY FEATURE EXTRACTION
# ============================================================

def extract_xray_features(pairs, encoder_b, encoder_c):
    """Extract 256D features and average across X-rays."""
    vectors = []

    for key, b_path, c_path in pairs:
        b_input = load_xray_tensor(
            b_path, f"{key} Model B"
        )
        c_input = load_xray_tensor(
            c_path, f"{key} Model C"
        )

        b_vector = np.asarray(
            encoder_b(
                b_input[None, ...],
                training=False,
            )
        ).reshape(-1)

        c_vector = np.asarray(
            encoder_c(
                c_input[None, ...],
                training=False,
            )
        ).reshape(-1)

        if b_vector.shape != (128,):
            raise ValueError("Invalid Model B features.")

        if c_vector.shape != (128,):
            raise ValueError("Invalid Model C features.")

        combined = np.concatenate([
            b_vector, c_vector
        ]).astype(np.float32)

        if not np.isfinite(combined).all():
            raise ValueError(
                f"{key}: invalid X-ray feature values."
            )

        vectors.append(combined)

    features = np.mean(
        np.stack(vectors),
        axis=0,
    ).astype(np.float32)

    if features.shape != (256,):
        raise ValueError("Invalid X-ray feature dimensions.")

    return features


# ============================================================
# 7. FUSION PREDICTION + PROCESSING TIMES
# ============================================================

def predict_fusion(ct_path, b_files, c_files):
    """Run frozen fusion and measure each inference stage."""
    start_total = time.perf_counter()

    if not ct_path:
        raise ValueError("Upload a CT volume.")

    pairs = pair_xray_files(b_files, c_files)

    # First run includes loading model checkpoints.
    start_load = time.perf_counter()
    assets = load_models()
    synchronize_gpu()
    load_time = time.perf_counter() - start_load

    # CT encoder time, including loading preprocessed input.
    synchronize_gpu()
    start_ct = time.perf_counter()

    ct_features, ct_probability = extract_ct_features(
        ct_path, assets["ct"]
    )

    synchronize_gpu()
    ct_time = time.perf_counter() - start_ct

    # X-ray encoder time, including all paired images.
    synchronize_gpu()
    start_xray = time.perf_counter()

    xray_features = extract_xray_features(
        pairs, assets["b"], assets["c"]
    )

    synchronize_gpu()
    xray_time = time.perf_counter() - start_xray

    # Frozen fusion classifier time.
    start_fusion = time.perf_counter()

    fusion_input = np.concatenate([
        ct_features, xray_features
    ]).astype(np.float64)[None, :]

    if fusion_input.shape != (1, 1280):
        raise ValueError("Fusion input must be 1280D.")

    probabilities = np.asarray(
        assets["fusion"].predict_proba(fusion_input)[0],
        dtype=float,
    )

    if (
        probabilities.shape != (2,)
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError("Invalid fusion probabilities.")

    p_negative = float(probabilities[0])
    p_positive = float(probabilities[1])

    fusion_time = time.perf_counter() - start_fusion

    positive = p_positive >= THRESHOLD

    label = (
        "COVID POSITIVE" if positive else "COVID NEGATIVE"
    )

    confidence = (
        p_positive if positive else p_negative
    )

    total_time = time.perf_counter() - start_total

    return {
        "label": label,
        "positive": positive,
        "confidence": confidence,
        "covid_positive": p_positive,
        "covid_negative": p_negative,
        "ct_positive": ct_probability,
        "ct_negative": 1.0 - ct_probability,
        "xray_count": len(pairs),
        "ct_time": ct_time,
        "xray_time": xray_time,
        "fusion_time": fusion_time,
        "load_time": load_time,
        "total_time": total_time,
    }


# ============================================================
# 8. CLASSIFICATION RESULT
# ============================================================

def classify(ct_path, b_files, c_files):
    """Generate four cards and the model information table."""
    try:
        r = predict_fusion(ct_path, b_files, c_files)

        label = html.escape(r["label"])

        # Classification card changes color by result.
        status = "positive" if r["positive"] else "negative"

        return f"""
        <div class="results-grid">

            <div class="result-card classification-{status}">
                <div class="result-heading">Classification</div>
                <div class="result-value">{label}</div>
            </div>

            <div class="result-card">
                <div class="result-heading">Confidence</div>
                <div class="result-value">
                    {r["confidence"]:.1%}
                </div>
            </div>

            <div class="result-card">
                <div class="result-heading">COVID Positive</div>
                <div class="result-value">
                    {r["covid_positive"]:.1%}
                </div>
            </div>

            <div class="result-card">
                <div class="result-heading">COVID Negative</div>
                <div class="result-value">
                    {r["covid_negative"]:.1%}
                </div>
            </div>

        </div>

        <div class="model-information">
            <h3>Model Information</h3>

            <table>
                <thead>
                    <tr>
                        <th>Information</th>
                        <th>CT Encoder</th>
                        <th>X-ray Encoder</th>
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
                        <td>{r["ct_positive"]:.1%}</td>
                        <td>{r["covid_positive"]:.1%}
                            (final fusion)</td>
                    </tr>

                    <tr>
                        <td>COVID Negative probability</td>
                        <td>{r["ct_negative"]:.1%}</td>
                        <td>{r["covid_negative"]:.1%}
                            (final fusion)</td>
                    </tr>

                    <tr>
                        <td>Processing time</td>
                        <td>{r["ct_time"]:.3f} s</td>
                        <td>{r["xray_time"]:.3f} s</td>
                    </tr>

                    <tr>
                        <td>Fusion classifier time</td>
                        <td colspan="2">
                            {r["fusion_time"]:.3f} s
                        </td>
                    </tr>

                    <tr>
                        <td>Model loading time</td>
                        <td colspan="2">
                            {r["load_time"]:.3f} s
                        </td>
                    </tr>

                    <tr>
                        <td>Total processing time</td>
                        <td colspan="2">
                            {r["total_time"]:.3f} s
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

    except Exception as exc:
        message = html.escape(
            f"{type(exc).__name__}: {exc}"
        )

        return (
            "<h3>Inference failed</h3>"
            f"<p>{message}</p>"
        )


# ============================================================
# 9. HOSPITAL-INSPIRED LIGHT THEME
# ============================================================

CSS = """
:root {
    color-scheme: light !important;
    --page: #f5f8fb;
    --surface: #ffffff;
    --text: #172033;
    --border: #cbd5e1;
    --medical-blue: #075985;
    --clinical-teal: #0d9488;
}

/* Page */
html, body, gradio-app, .gradio-container {
    background: var(--page) !important;
    color: var(--text) !important;
    color-scheme: light !important;
}

.gradio-container {
    max-width: none !important;
    width: 100% !important;
    padding: 12px 24px !important;
}

/* TITLE CARD — EDIT FONT SIZES HERE */
.hospital-header {
    background: var(--medical-blue) !important;
    border-radius: 10px !important;
    padding: 14px 20px !important;
    margin-bottom: 10px !important;
}

.hospital-header,
.hospital-header * {
    color: #ffffff !important;
}

.hospital-header h1 {
    font-size: 25px !important; /* Main title */
    margin: 0 !important;
}

.hospital-header p {
    font-size: 14px !important; /* Subtitle */
    margin: 3px 0 0 !important;
}

/* Input panels */
.clinical-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: 3px solid var(--clinical-teal) !important;
    border-radius: 9px !important;
    padding: 12px !important;
}

/* RUN BUTTON — EDIT FONT SIZE HERE */
#run-button {
    background: #d7f0f7 !important;
    color: var(--medical-blue) !important;
    border: 1px solid #9bd3e3 !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    min-height: 40px !important;
}

#run-button:hover {
    background: #bfe5f0 !important;
}

/* Output container */
.clinical-output {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    padding: 12px !important;
}

/* Four equal-width result cards */
.results-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    width: 100%;
}

.result-card {
    min-width: 0;
    background: #f8fafc;
    border: 1px solid var(--border);
    border-top: 3px solid var(--clinical-teal);
    border-radius: 9px;
    padding: 14px 8px;
    text-align: center;
}

/* Classification — COVID POSITIVE */
.result-card.classification-positive {
    background: #fff1f2;
    border: 1px solid #fda4af;
    border-top: 3px solid #dc2626;
}

.result-card.classification-positive .result-value {
    color: #b91c1c;
}

/* Classification — COVID NEGATIVE */
.result-card.classification-negative {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-top: 3px solid #16a34a;
}

.result-card.classification-negative .result-value {
    color: #15803d;
}

/* RESULT CARDS — EDIT FONT SIZES HERE */
.result-heading {
    font-size: 15px; /* Card headings */
    font-weight: 600;
    color: #64748b;
    margin-bottom: 6px;
}

.result-value {
    font-size: 28px; /* Main result values */
    font-weight: 700;
    color: var(--medical-blue);
    overflow-wrap: anywhere;
}

/* Model information */
.model-information {
    margin-top: 12px;
    width: 100%;
}

.model-information h3 {
    font-size: 18px; /* Table section title */
    color: var(--medical-blue);
    margin: 0 0 8px;
}

/* MODEL TABLE — EDIT FONT SIZES HERE */
.model-information table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: 15px; /* Table body */
    color: var(--text);
}

.model-information th {
    background: #edf5fa;
    color: var(--medical-blue);
    font-size: 15px; /* Table headings */
    font-weight: 700;
    text-align: left;
}

.model-information th,
.model-information td {
    padding: 9px 10px;
    border: 1px solid var(--border);
    overflow-wrap: anywhere;
}

.model-information th:first-child,
.model-information td:first-child {
    width: 34%;
}

.model-information th:nth-child(2),
.model-information td:nth-child(2),
.model-information th:nth-child(3),
.model-information td:nth-child(3) {
    width: 33%;
}

/* FOOTER — EDIT FONT SIZE HERE */
.clinical-footer {
    font-size: 12px !important;
    color: #64748b !important;
}

/* Responsive: 4 cards become 2 x 2 */
@media (max-width: 900px) {
    .results-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .result-value {
        font-size: 22px;
    }

    .gradio-container {
        padding: 8px !important;
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
# 10. GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    theme=gr.themes.Default(),
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

    # Update previews automatically when files are uploaded.
    ct_input.change(
        fn=preview_ct,
        inputs=ct_input,
        outputs=ct_preview,
    )

    xray_b_input.change(
        fn=preview_xray,
        inputs=xray_b_input,
        outputs=xray_b_preview,
    )

    xray_c_input.change(
        fn=preview_xray,
        inputs=xray_c_input,
        outputs=xray_c_preview,
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
        fn=classify,
        inputs=[ct_input, xray_b_input, xray_c_input],
        outputs=output,
    )

    gr.Markdown(
        "Research prototype. Inputs must belong to the "
        "same patient. Not for clinical diagnosis.",
        elem_classes="clinical-footer",
    )


# ============================================================
# 11. START APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch(inbrowser=True)
