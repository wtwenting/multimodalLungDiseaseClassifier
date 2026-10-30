"""Raw CT + X-ray fusion demonstration (research only, not clinical use).

Source-of-truth research functions imported from ensemble/src:
- prepare_ct_v5_final.process_source: CT V5 preprocessing/QC
- clean_modelc_xray: X-ray robust/resize/crop and reading
- audit_adaptive_xray: image quality decisions
- compare_encoders.model_b_pixels: X-ray Model B preprocessing
- ct_v61_frozen_baseline.load_checkpoint: frozen CT architecture

All paths resolve relative to the repository root. No research script is run
as a command, and no retraining occurs. Uncertain X-rays fail closed.
"""

import html
import logging
import traceback
import json
import os
import sys
import tempfile
import time
import webbrowser
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("fusion_gui")

# The model packages cache downloaded weights; packages alone do not bundle them.
# Respect user-defined caches; otherwise use framework defaults.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "ensemble" / "src"
HANDOVER = ROOT / "ensemble" / "output" / "handover" / "CT_Xray_GUI_Handover"
MODELS = HANDOVER / "models"
CONFIG = HANDOVER / "config"
if not SRC.is_dir():
    raise FileNotFoundError(f"Missing research source folder: {SRC}")
sys.path.insert(0, str(SRC))

import cv2
import gradio as gr
import joblib
import nibabel as nib
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from lungmask import LMInferer

from audit_adaptive_xray import classify as classify_quality, stats as image_stats
from clean_modelc_xray import crop64, read_image, resize224, robust
from compare_encoders import model_b_pixels
from ct_v61_frozen_baseline import load_checkpoint
from prepare_ct_v5_final import process_source

CT_MODEL = MODELS / "ct" / "v61_13a378082055_best.pt"
XRAY_B = MODELS / "xray" / "model_B_v2_27sep.keras"
XRAY_C = MODELS / "xray" / "modelC_resnet50_18oct.keras"
FUSION_MODEL = MODELS / "fusion" / "ct_b_xray_logistic_frozen.joblib"
SIDE_MODEL = MODELS / "side_detector" / "side_clf.joblib"
INVERT_TEMPLATE = MODELS / "side_detector" / "invert_template.npy"
THRESHOLDS = CONFIG / "xray_thresholds_train_only.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CT_VIEWS = 12
THRESHOLD = 0.5
# Verified from ModelB_CovidvsNormal.ipynb (label_map) and
# BIMCV_ModelC_resnet50.ipynb (manifest class/label output):
# 0 = COVID Negative, 1 = COVID Positive.
XRAY_B_SIGMOID_COVID_POSITIVE = True
XRAY_C_SIGMOID_COVID_POSITIVE = True

# X-ray quality control
ENABLE_XRAY_QC = False  # False: report QC but do not reject; True: require ACCEPT

def required(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return path


def sync_cuda():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@lru_cache(maxsize=1)
def load_lungmask():
    """Reuse one R231 segmenter for all CT uploads."""
    return LMInferer(modelname="R231", fillmodel=None, tqdm_disable=True)


@lru_cache(maxsize=1)
def load_xray_qc():
    """Reuse frozen X-ray side detector, template and MobileNetV2."""
    thresholds = json.loads(required(THRESHOLDS).read_text(encoding="utf-8"))
    if set(thresholds) != {"mean_low", "mean_high", "std_low", "dark_high"}:
        raise ValueError("Unexpected X-ray threshold configuration")
    template = np.load(required(INVERT_TEMPLATE), allow_pickle=False)
    if template.shape != (64, 64):
        raise ValueError("Inversion template must be 64x64")
    side_clf = joblib.load(required(SIDE_MODEL))  # Trusted project artifact only.
    if 1 not in side_clf.classes_:
        raise ValueError("Side detector missing class 1")
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False, pooling="avg", input_shape=(224, 224, 3),
        weights="imagenet",
    )
    backbone.trainable = False
    return thresholds, template, side_clf, backbone


def build_xray_outputs(full_model, name):
    """Reuse original Sequential weights; return 128D features and logits/probs."""
    x_in = tf.keras.Input(shape=(224, 224, 3), name=f"{name}_input")
    x, embedding = x_in, None
    for layer in full_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x, training=False)
        if layer.name == "dense":
            embedding = x
    if embedding is None or embedding.shape[-1] != 128:
        raise ValueError(f"{name}: expected named dense layer with 128 features")
    model = tf.keras.Model(x_in, [embedding, x], name=f"{name}_inference")
    model.trainable = False
    return model


@lru_cache(maxsize=1)
def load_models():
    """Models loaded once per Python process, reused on every request."""
    ct, cfg = load_checkpoint(required(CT_MODEL), DEVICE)
    if (cfg.get("arch"), cfg.get("pooling"), cfg.get("views")) != ("densenet121", "mean", 12):
        raise ValueError("Unexpected CT checkpoint configuration")
    b = build_xray_outputs(tf.keras.models.load_model(str(required(XRAY_B)), compile=False), "model_b")
    c = build_xray_outputs(tf.keras.models.load_model(str(required(XRAY_C)), compile=False), "model_c")
    fusion = joblib.load(required(FUSION_MODEL))  # Trusted project artifact only.
    if list(fusion.named_steps) != ["scale", "clf"]:
        raise ValueError("Unexpected frozen fusion pipeline")
    if fusion.named_steps["scale"].n_features_in_ != 1280 or list(fusion.classes_) != [0, 1]:
        raise ValueError("Fusion requires 1280D and classes [0,1]")
    return ct, b, c, fusion


def preprocess_ct(path):
    """Exact source process_source() plus its original PASS/REVIEW QC."""
    start = time.perf_counter()
    if not path or not str(path).lower().endswith(".nii.gz"):
        raise ValueError("Upload one CT .nii.gz file")
    volume, qc = process_source(required(Path(path)), load_lungmask())
    if qc["status"] != "PASS":
        raise ValueError(
            f"CT QC {qc['status']}: lung retention={qc['mask_retention_estimate']:.3f}; "
            f"boundary fraction={qc['mask_boundary_fraction']:.3f}. Review scan coverage."
        )
    volume = np.asarray(volume, dtype=np.float32)
    if volume.shape != (64, 192, 192) or not np.isfinite(volume).all():
        raise ValueError("Invalid CT V5 tensor")
    return volume, time.perf_counter() - start


def xray_qc(raw):
    """Follow extract_xray_test_features_reviewed.py side/inversion workflow."""
    thresholds, template, clf, backbone = load_xray_qc()
    candidate = resize224(robust(raw))
    # Original crop64 helper accepts a file path.
    with tempfile.TemporaryDirectory() as folder:
        staged = Path(folder) / "candidate.png"
        Image.fromarray(candidate).save(staged)
        content_fraction, crop = crop64(staged)
    inverted = float((crop * template).mean()) < 0
    mobile = np.repeat(candidate[..., None], 3, axis=-1).astype(np.float32)
    if inverted:
        mobile = 255.0 - mobile
    mobile = tf.keras.applications.mobilenet_v2.preprocess_input(mobile)
    embedding = np.asarray(backbone(mobile[None], training=False))
    side_probability = float(clf.predict_proba(embedding)[0, list(clf.classes_).index(1)])
    quality = classify_quality(image_stats(raw), thresholds)
    decision = {
        "normal": "direct", "dark": "robust_1_99", "bright": "robust_1_99",
        "low_contrast": "clahe_after_robust",
    }[quality]
    review = 0.2 <= side_probability <= 0.8 or inverted or content_fraction < 0.6
    status = "EXCLUDE_SIDE" if side_probability > 0.5 else "REVIEW" if review else "ACCEPT"
    return status, decision, inverted, side_probability, content_fraction


def preprocess_xray(path):
    """One raw PNG -> Model B/C tensors; optionally enforce QC."""
    start = time.perf_counter()
    if not path or Path(path).suffix.lower() != ".png":
        raise ValueError("X-ray input must be .png")
    raw = read_image(required(Path(path)))
    status, decision, inverted, side_p, fraction = xray_qc(raw)
    if status != "ACCEPT":
        details = (
            f"X-ray QC {status}: {Path(path).name}, side={side_p:.3f}, "
            f"inverted={inverted}, content={fraction:.3f}."
        )
        if ENABLE_XRAY_QC:
            raise ValueError(details + " Manual review required.")
        LOG.warning("QC enforcement disabled; proceeding despite %s", details)
    # Model B: compare_encoders.model_b_pixels, then TF resize/pad.
    b_pixels = model_b_pixels(
        SimpleNamespace(orig_path=str(path), invert_final=inverted),
        {"selective_clahe_decision": decision}, cv2,
    )
    b = tf.image.resize_with_pad(
        tf.convert_to_tensor(b_pixels[..., None], tf.float32),
        224, 224, antialias=False,
    )
    b = np.asarray(tf.repeat(b, 3, axis=-1), dtype=np.float32) / 255.0
    # Model C: clean_modelc_xray.robust + inversion before resize.
    c_pixels = robust(raw)
    if inverted:
        c_pixels = 255 - c_pixels
    c_pixels = resize224(c_pixels)
    c = np.repeat(c_pixels[..., None], 3, axis=-1).astype(np.float32) / 255.0
    for name, tensor in (("B", b), ("C", c)):
        if tensor.shape != (224, 224, 3) or not np.isfinite(tensor).all():
            raise ValueError(f"Invalid X-ray {name} tensor")
    return b, c, time.perf_counter() - start


def extract_ct(volume, model):
    centers = torch.linspace(3, 60, CT_VIEWS).round().long().tolist()
    x = torch.from_numpy(np.ascontiguousarray(volume))
    views = torch.stack([x[i - 1:i + 2] for i in centers]).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        features, logits = model.features_and_logits(views)
        probability = float(logits.float().softmax(-1)[0, 1].cpu())
    vector = features[0].float().cpu().numpy().astype(np.float32)
    if vector.shape != (1024,) or not np.isfinite(vector).all():
        raise ValueError("Invalid CT feature vector")
    return vector, probability


def extract_xray(b_tensor, c_tensor, model_b, model_c):
    """Return frozen 256D features and diagnostic B/C soft-vote probability."""
    b_feat, b_output = model_b(b_tensor[None], training=False)
    c_feat, c_output = model_c(c_tensor[None], training=False)
    b = np.asarray(b_feat).reshape(-1)
    c = np.asarray(c_feat).reshape(-1)
    if b.shape != (128,) or c.shape != (128,):
        raise ValueError("X-ray encoder must return 128D each")
    features = np.concatenate((b, c)).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Invalid X-ray feature vector")

    def sigmoid_probability(output, name, positive_mapping):
        values = np.asarray(output).reshape(-1)
        if values.size != 1 or not np.isfinite(values).all():
            raise ValueError(f"{name}: expected one finite sigmoid probability")
        p = float(values[0])
        if not 0 <= p <= 1:
            raise ValueError(f"{name}: sigmoid probability outside [0, 1]")
        return p if positive_mapping else 1.0 - p

    p_b = sigmoid_probability(b_output, "Model B", XRAY_B_SIGMOID_COVID_POSITIVE)
    p_c = sigmoid_probability(c_output, "Model C", XRAY_C_SIGMOID_COVID_POSITIVE)
    return features, (p_b + p_c) / 2.0


def predict(ct_path, xray_path, report=None):
    """One CT and one X-ray, both from the same patient (user responsibility)."""
    if not ct_path or not xray_path:
        raise ValueError("Both CT and X-ray inputs are required")
    start_total = time.perf_counter()
    if report: report("Loading frozen models")
    start = time.perf_counter()
    ct_model, b_model, c_model, fusion = load_models()
    sync_cuda()
    load_time = time.perf_counter() - start

    if report: report("Preprocessing CT volume and checking lung coverage")
    volume, ct_prep = preprocess_ct(ct_path)
    if report: report("Preprocessing X-ray and checking image quality")
    b_tensor, c_tensor, xr_prep = preprocess_xray(xray_path)

    sync_cuda()
    start = time.perf_counter()
    if report: report("Extracting CT features")
    ct_features, ct_positive = extract_ct(volume, ct_model)
    sync_cuda()
    ct_time = time.perf_counter() - start

    start = time.perf_counter()
    if report: report("Extracting X-ray Model B and C features")
    xr_features, xr_positive = extract_xray(b_tensor, c_tensor, b_model, c_model)
    xr_time = time.perf_counter() - start

    start = time.perf_counter()
    if report: report("Running frozen fusion classifier")
    vector = np.concatenate((ct_features, xr_features)).astype(np.float64)[None, :]
    probs = np.asarray(fusion.predict_proba(vector)[0], dtype=float)
    if probs.shape != (2,) or not np.isfinite(probs).all():
        raise ValueError("Invalid fusion probabilities")
    p_negative, p_positive = map(float, probs)
    fusion_time = time.perf_counter() - start
    positive = p_positive >= THRESHOLD
    return dict(
        label="COVID POSITIVE" if positive else "COVID NEGATIVE",
        positive=positive, confidence=p_positive if positive else p_negative,
        ct_positive=ct_positive, xr_positive=xr_positive, ct_prep=ct_prep, xr_prep=xr_prep,
        ct_time=ct_time, xr_time=xr_time, fusion_time=fusion_time,
        load_time=load_time, total_time=time.perf_counter() - start_total,
    )


def classify(ct_path, xray_path):
    stage = "Checking uploads"
    try:
        def report(message):
            nonlocal stage
            stage = message
            LOG.info("Inference stage: %s", stage)

        r = predict(ct_path, xray_path, report)
        status = "positive" if r["positive"] else "negative"
        cards = f"""
        <div class="results-grid">
          <div class="result-card classification-{status}">
            <div class="result-heading">Classification</div>
            <div class="result-value">{r['label']}</div>
          </div>
          <div class="result-card">
            <div class="result-heading">Confidence</div>
            <div class="result-value">{r['confidence']:.0%}</div>
          </div>
        </div>"""
        table = f"""
        <div class="model-information"><table>
          <thead><tr><th>Information</th><th>CT Encoder</th><th>X-ray Encoders</th></tr></thead>
          <tbody>
          <tr><td>Architecture</td><td>DenseNet121 V6.1 Mean</td><td>Model B + Model C</td></tr>
          <tr><td>Images processed</td><td>12 three-slice views</td><td>1 X-ray</td></tr>
          <tr><td>Features extracted</td><td>1024D</td><td>256D (128D + 128D)</td></tr>
          <tr><td>COVID Positive probability</td><td>{r['ct_positive']:.0%}</td><td>{r['xr_positive']:.0%}</td></tr>
          <tr><td>COVID Negative probability</td><td>{1-r['ct_positive']:.0%}</td><td>{1-r['xr_positive']:.0%}</td></tr>
          <tr><td>Preprocessing time</td><td>{r['ct_prep']:.1f} s</td><td>{r['xr_prep']:.1f} s</td></tr>
          <tr><td>Encoder processing time</td><td>{r['ct_time']:.1f} s</td><td>{r['xr_time']:.1f} s</td></tr>
          <tr><td>Fusion classifier time</td><td colspan="2">{r['fusion_time']:.1f} s</td></tr>
          <tr><td>Model loading time</td><td colspan="2">{r['load_time']:.1f} s</td></tr>
          <tr><td>Total processing time</td><td colspan="2">{r['total_time']:.1f} s</td></tr>
          </tbody></table>
          <p class="model-note">X-ray diagnostic percentages use equal-weight Model B/C sigmoid outputs (COVID-positive class mapping verified against training notebooks). Final confidence comes from the frozen CT + X-ray fusion classifier. Research use only.</p>
        </div>"""
        return cards, table
    except Exception as exc:
        LOG.exception("Inference failed at: %s", stage)
        detail = html.escape(str(exc) or type(exc).__name__)
        return (f'<div class="error-panel"><strong>Inference stopped</strong>'
                f'<p><b>Stage:</b> {html.escape(stage)}</p>'
                f'<p><b>Reason:</b> {detail}</p>'
                f'<p>See the terminal for the full traceback. The application remains running.</p></div>', "")


def preview_ct(path):
    """Preview an axial slice from a raw .nii.gz CT (display only)."""
    if not path:
        return None, ""
    try:
        if not Path(path).name.lower().endswith(".nii.gz"):
            raise ValueError("Expected a raw CT .nii.gz file.")

        scan = nib.as_closest_canonical(nib.load(str(path)))
        if len(scan.shape) != 3 or min(scan.shape) < 2:
            raise ValueError(f"Expected a 3D CT volume; received {scan.shape}.")

        # Read only the middle axial slice; avoid loading the entire CT volume.
        middle = scan.shape[2] // 2
        pixels = np.asarray(scan.dataobj[:, :, middle], dtype=np.float32)
        if not np.isfinite(pixels).all():
            raise ValueError("CT preview contains non-finite pixel values.")

        # Standard lung window; display does not modify the inference input.
        pixels = np.clip(pixels, -1000.0, 400.0)
        pixels = np.rint((pixels + 1000.0) * (255.0 / 1400.0)).astype(np.uint8)
        preview = Image.fromarray(np.rot90(pixels))
        return preview, f"CT preview ready: axial slice {middle + 1}/{scan.shape[2]}."
    except Exception as exc:
        LOG.exception("CT preview failed")
        return None, f"CT preview failed: {type(exc).__name__}: {exc}"


def preview_xray(path):
    """Preview a raw PNG X-ray, including 16-bit grayscale PNGs."""
    if not path:
        return None, ""
    try:
        if Path(path).suffix.lower() != ".png":
            raise ValueError("Expected a raw X-ray .png file.")

        with Image.open(path) as source:
            source.load()
            if min(source.size) < 2:
                raise ValueError("X-ray dimensions are too small.")
            if source.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
                pixels = np.asarray(source, dtype=np.float32)
                if not np.isfinite(pixels).all():
                    raise ValueError("X-ray contains non-finite pixel values.")
                low, high = np.percentile(pixels, [1, 99])
                if high <= low:
                    low, high = float(pixels.min()), float(pixels.max())
                pixels = np.clip((pixels - low) * 255.0 / max(high - low, 1.0), 0, 255)
                preview = Image.fromarray(pixels.astype(np.uint8)).convert("RGB")
            else:
                preview = source.convert("RGB").copy()
        return preview, "X-ray PNG preview ready."
    except Exception as exc:
        LOG.exception("X-ray preview failed")
        return None, f"X-ray preview failed: {type(exc).__name__}: {exc}"


LIGHT_THEME = gr.themes.Default(primary_hue="blue", secondary_hue="teal", neutral_hue="slate")
CSS = """
.gradio-container {max-width:none!important;width:100%!important;padding:12px 24px!important}
.hospital-header {background:#075985;border-radius:10px;padding:14px 20px;margin-bottom:10px}
.hospital-header,.hospital-header * {color:white!important}
.hospital-header h1 {font-size:25px!important;margin:0}
.hospital-header p {font-size:14px!important;margin:3px 0 0}
.clinical-panel {border:1px solid #CBD5E1;border-top:3px solid #0D9488;border-radius:9px;padding:12px}
#run-button {background:#D7F0F7!important;color:#075985!important;border:1px solid #9BD3E3!important;font-size:16px!important;font-weight:700;min-height:40px}
#run-button:hover {background:#BFE5F0!important}
.results-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.result-card {min-width:0;background:#F8FAFC;border:1px solid #CBD5E1;border-top:3px solid #0D9488;border-radius:9px;padding:14px 8px;text-align:center}
.classification-positive {background:#FFF1F2;border-color:#FDA4AF;border-top-color:#DC2626}
.classification-negative {background:#F0FDF4;border-color:#86EFAC;border-top-color:#16A34A}
.classification-positive .result-value {color:#B91C1C}
.classification-negative .result-value {color:#15803D}
.result-heading {font-size:15px;font-weight:600;color:#64748B;margin-bottom:6px}
.result-value {font-size:28px;font-weight:700;color:#075985;overflow-wrap:anywhere}
.model-information table {width:100%;border-collapse:collapse;table-layout:fixed;font-size:15px;background:white;color:#172033}
.model-information th {background:#EDF5FA;color:#075985;font-weight:700;text-align:left}
.model-information td {background:white;color:#172033}
.model-information th,.model-information td {padding:9px 10px;border:1px solid #CBD5E1;overflow-wrap:anywhere}
.model-information th:first-child {width:34%}
.model-information th:nth-child(2),.model-information th:nth-child(3) {width:33%}
.model-note,.clinical-footer {font-size:12px;color:#64748B}
.error-panel {background:#FFF1F2;border:1px solid #FDA4AF;border-radius:8px;padding:12px;color:#7F1D1D}
.error-panel p {color:#7F1D1D}
.upload-status {font-size:12px;color:#475569}
@media(max-width:900px) {.gradio-container{padding:8px!important}.result-value{font-size:22px}.model-information table,.model-information th{font-size:13px}.model-information th,.model-information td{padding:6px}}
"""

with gr.Blocks(theme=LIGHT_THEME, css=CSS, title="Multimodal Lung Disease Classifier") as demo:
    gr.Markdown("# Multimodal Lung Disease Classifier\nCT + X-ray | Frozen Fusion Model", elem_classes="hospital-header")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=250, elem_classes="clinical-panel"):
            ct_input = gr.File(label="CT input (.nii.gz)", file_types=[".nii.gz"], type="filepath", file_count="single")
            ct_preview = gr.Image(label="CT Preview", interactive=False, height=240)
            ct_status = gr.Textbox(label="CT upload status", interactive=False, lines=1, elem_classes="upload-status")
        with gr.Column(scale=1, min_width=250, elem_classes="clinical-panel"):
            xray_input = gr.File(label="X-ray input (.png)", file_types=[".png"], type="filepath", file_count="single")
            xray_preview = gr.Image(label="X-ray Preview", interactive=False, height=240)
            xray_status = gr.Textbox(label="X-ray upload status", interactive=False, lines=1, elem_classes="upload-status")
    ct_input.change(preview_ct, inputs=ct_input, outputs=[ct_preview, ct_status])
    xray_input.change(preview_xray, inputs=xray_input, outputs=[xray_preview, xray_status])
    run_button = gr.Button("Run Fusion Classification", elem_id="run-button")
    summary = gr.HTML("Upload one CT volume and one X-ray from the same patient.")
    with gr.Accordion("Model Information", open=False):
        information = gr.HTML()
    run_button.click(classify, inputs=[ct_input, xray_input], outputs=[summary, information], concurrency_limit=1)
    gr.Markdown("Research prototype. Both images must belong to the same patient. Not for clinical diagnosis.", elem_classes="clinical-footer")

if __name__ == "__main__":
    try:
        app, local_url, _ = demo.launch(
            inbrowser=False, server_name="127.0.0.1", server_port=None,
            prevent_thread_lock=True, show_error=True,
        )
        webbrowser.open(f"{local_url.rstrip('/')}?__theme=light")
        demo.block_thread()
    except KeyboardInterrupt:
        LOG.info("GUI stopped by user")
    except Exception:
        LOG.exception("Gradio startup failed")
        raise

