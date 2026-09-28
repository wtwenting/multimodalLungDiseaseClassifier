import gradio as gr
import torch
import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from torch import nn
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

GUI_DIR = Path(__file__).resolve().parent
ROOT_DIR = GUI_DIR.parent

MODEL_PATH_CT = ROOT_DIR / "CT" / "model" / "ct_2d_covid_normal_resnet18.pth"
MODEL_PATH_AUDIO = ROOT_DIR / "cough" / "data" / "coughvid_20211012_processed" / "models" / "moderate_sampling_v4.keras"
MODEL_PATH_XRAY = ROOT_DIR / "xraydata" / "Models" / "ModelB_CovidvsNormal_27sep" / "model_B_v2_27sep.keras"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Temporary audio assumptions - update from training notebook when available
SAMPLE_RATE_AUDIO = 22050
N_MELS_AUDIO = 64
TIME_FRAMES_AUDIO = 688
CLASS_INDEX_NORMAL_AUDIO = 0
CLASS_INDEX_COVID_AUDIO = 1


# ============================================================
# CT MODEL
# Must match training: ResNet18, NORMAL=0, COVID=1
# ============================================================

ct_model = models.resnet18(weights=None)
ct_model.fc = nn.Linear(ct_model.fc.in_features, 2)
ct_model.load_state_dict(torch.load(MODEL_PATH_CT, map_location=DEVICE, weights_only=True))
ct_model.to(DEVICE).eval()

ct_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_ct(image):
    """Run CT model and return class probabilities."""
    x = ct_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(ct_model(x), dim=1)[0]
    return {"NORMAL": float(probs[0]), "COVID": float(probs[1])}


# ============================================================
# AUDIO MODEL
# Keras CNN: input (64, 688, 1), output 3 classes
# ============================================================

audio_model = tf.keras.models.load_model(MODEL_PATH_AUDIO, compile=False)

def preprocess_audio(audio_path):
    """Convert uploaded audio into model-ready log-mel input."""
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE_AUDIO, mono=True)
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE_AUDIO, n_mels=N_MELS_AUDIO)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] < TIME_FRAMES_AUDIO:
        log_mel = np.pad(log_mel, ((0, 0), (0, TIME_FRAMES_AUDIO - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :TIME_FRAMES_AUDIO]

    return log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)

def predict_audio(audio):
    """Run 3-class audio model and return normalized NORMAL/COVID probabilities."""
    probs = audio_model.predict(preprocess_audio(audio), verbose=0)[0]
    normal = float(probs[CLASS_INDEX_NORMAL_AUDIO])
    covid = float(probs[CLASS_INDEX_COVID_AUDIO])
    total = normal + covid
    if total > 0:
        normal, covid = normal / total, covid / total
    return {"NORMAL": normal, "COVID": covid}


# ============================================================
# XRAY MODEL
# Must match training: 224x224 RGB, 0-1 scaling, NORMAL=0, COVID=1
# ============================================================

xray_model = tf.keras.models.load_model(MODEL_PATH_XRAY, compile=False)

def preprocess_xray(image):
    """Match X-ray training preprocessing: RGB, resize-with-pad, scale to 0-1."""
    x = tf.convert_to_tensor(np.asarray(image.convert("RGB")))
    x = tf.image.resize_with_pad(x, 224, 224)
    x = tf.cast(x, tf.float32) / 255.0
    return x[tf.newaxis, ...]

def predict_xray(image):
    """Run X-ray model; sigmoid output is P(COVID)."""
    covid = float(xray_model.predict(preprocess_xray(image), verbose=0)[0][0])
    return {"NORMAL": 1.0 - covid, "COVID": covid}


# ============================================================
# MULTIMODAL PREDICTION
# ============================================================

def predict(audio, xray, ct):
    if audio is None and xray is None and ct is None:
        return '<div class="empty">Please provide at least one input.</div>'

    results = {}
    if audio is not None:
        results["Audio"] = predict_audio(audio)
    if xray is not None:
        results["X-ray"] = predict_xray(xray)
    if ct is not None:
        results["CT"] = predict_ct(ct)

    normal = sum(r["NORMAL"] for r in results.values()) / len(results)
    covid = sum(r["COVID"] for r in results.values()) / len(results)

    if covid > normal:
        label, confidence, style = "COVID", covid, "covid"
    else:
        label, confidence, style = "NORMAL", normal, "normal"

    models_used = " + ".join(results)
    return f"""
    <div class="result {style}">
        <div class="result-label">{label}</div>
        <div class="confidence">{confidence:.1%}</div>
        <div class="details">
            COVID: {covid:.1%} &nbsp; | &nbsp; NORMAL: {normal:.1%}
            <br>Model(s): {models_used}
        </div>
    </div>
    """


# ============================================================
# UI STYLE
# ============================================================

css = """

/* Use almost the full browser width */
.gradio-container {
    width: 98vw !important;
    max-width: none !important;
    margin: 0 auto !important;
    padding: 6px 12px !important;
}

/* Compact title */
.title {
    text-align: center;
    margin: 0 !important;
}

.title h1 {
    margin: 0 !important;
    font-size: 26px !important;
}

.title p {
    margin: 2px 0 6px !important;
}

/* Three equal full-width input panels */
.input-row {
    width: 100% !important;
    gap: 12px !important;
    flex-wrap: nowrap !important;
}

.input-box {
    flex: 1 1 33.33% !important;
    width: 33.33% !important;
    min-width: 0 !important;
    border: 1px solid #d0d0d0;
    border-radius: 10px;
    padding: 8px !important;
}

.input-box h2 {
    margin: 0 0 4px !important;
    font-size: 19px !important;
}

/* Run button */
#run-button {
    margin-top: 4px !important;
    min-height: 38px !important;
}

/* Output */
.output-title h2 {
    margin: 5px 0 !important;
    font-size: 19px !important;
}

.result {
    text-align: center;
    padding: 14px;
    border-radius: 10px;
}

.result-label {
    font-size: 34px;
    font-weight: 700;
}

.confidence {
    font-size: 23px;
    font-weight: 600;
}

.details {
    font-size: 14px;
    margin-top: 6px;
}

/* COVID result */
.result.covid {
    background: #ffe6e6 !important;
    border: 2px solid #cc0000 !important;
}

.result.covid,
.result.covid * {
    color: #990000 !important;
}

/* NORMAL result */
.result.normal {
    background: #e4f7e8 !important;
    border: 2px solid #159447 !important;
}

.result.normal,
.result.normal * {
    color: #08752f !important;
}

/* Dev mode: fixed bottom-right toggle */
#dev-mode {
    position: fixed !important;
    right: 16px !important;
    bottom: 12px !important;
    z-index: 9999 !important;
    width: auto !important;
}

/* Empty output */
.empty {
    text-align: center;
    padding: 14px;
    border: 1px dashed #aaa;
    border-radius: 10px;
    color: #555 !important;
}

"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(css=css) as demo:

    gr.Markdown(
        """
        # Multimodal Lung Disease Classifier
        Upload one or more available inputs.
        """,
        elem_classes="title"
    )

    # Three inputs across full browser width
    with gr.Row(equal_height=True, elem_classes="input-row"):

        with gr.Column(scale=1, min_width=0, elem_classes="input-box"):
            gr.Markdown("## Audio")
            audio_input = gr.Audio(
                type="filepath",
                label="Respiratory Audio"
            )

        with gr.Column(scale=1, min_width=0, elem_classes="input-box"):
            gr.Markdown("## X-ray")
            xray_input = gr.Image(
                type="pil",
                label="Chest X-ray",
                height=170
            )

        with gr.Column(scale=1, min_width=0, elem_classes="input-box"):
            gr.Markdown("## CT")
            ct_input = gr.Image(
                type="pil",
                label="CT Image",
                height=170
            )

    # Run classification
    predict_button = gr.Button(
        "Run Classification",
        variant="primary",
        elem_id="run-button"
    )

    # Output area
    gr.Markdown(
        "## Classification Output",
        elem_classes="output-title"
    )

    output = gr.HTML(
        '<div class="empty">Upload an input and click Run Classification.</div>'
    )

    predict_button.click(
        fn=predict,
        inputs=[audio_input, xray_input, ct_input],
        outputs=output
    )

    dev_mode = gr.Checkbox(
        label="Dev mode",
        value=False,
        elem_id="dev-mode"
    )

    # Dev mode currently toggles light/dark theme; extend here later.
    dev_mode.change(
        fn=None,
        inputs=dev_mode,
        outputs=None,
        js="""(enabled) => {
            document.documentElement.classList.toggle('dark', enabled);
            return [];
        }"""
    )


# ============================================================
# START
# ============================================================

demo.launch(inbrowser=True)