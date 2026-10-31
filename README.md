# Multimodal Lung Disease Classifier

An academic research and demonstration project investigating COVID-19 classification from respiratory cough audio, chest X-rays, and chest CT images. The repository includes separate modality experiments, a **patient-matched CT + X-ray frozen-feature fusion** workflow, and a local GUI demonstration.

> **Research only.** The models are not clinically validated diagnostic tools. Dataset labels and patient matching do not establish clinical diagnosis or same-session imaging.

## Repository guide

| Directory / file | Purpose | Further information |
|---|---|---|
| `cough/` | COUGHVID and Coswara audio preparation, exploratory analysis, CNN/LSTM experiments and Google HeAR features | [Cough README](cough/README.md) |
| `CT/` | CT model training and associated source/checkpoints | See source files under `CT/` |
| `xraydata/` | Chest X-ray data-processing and model workstream | [X-ray scripts README](xraydata/scripts/README.md) |
| `ensemble/` | CT + X-ray feature alignment, baselines, frozen fusion, evaluation, uncertainty and modality analyses | [Ensemble README](ensemble/README.md) |
| `GUI/` | GUI application and integration scripts | `GUI/GUI_demo.py` |
| `Demo/Patient_Data_Preprocessed/` | Five previously evaluated patients' encoder-ready arrays and reference features | [Preprocessed demo README](Demo/Patient_Data_Preprocessed/README.md) |
| `Demo/Patient_Data_Raw/` | Raw demo inputs where available, plus manifests and reference features | [Raw demo README](Demo/Patient_Data_Raw/README.md) |
| `1.Setup_and_Run.sh` | macOS/Linux environment setup and GUI launcher | Run from repository root |
| `1.Setup_and_Run.cmd` | Windows environment setup and GUI launcher | Run from Windows Command Prompt |
| `requirements.txt` | Top-level GUI and project dependency declarations | Review platform compatibility before installing |

## Project workflow

1. **Respiratory audio:** Analyze COUGHVID and Coswara cough recordings, compare preprocessing and classifiers, and evaluate Google HeAR representations. This is a separate research workstream; the documented selected CT + X-ray fusion model does **not** include cough features.
2. **CT and chest X-ray:** Prepare medical images and generate modality-specific representations using the corresponding model pipelines.
3. **CT + X-ray fusion:** For the selected later-phase model, concatenate 1,024-dimensional frozen CT Encoder B features with 256-dimensional frozen X-ray Model B + C features (CT first), then apply the fitted `StandardScaler` and Logistic Regression classifier at threshold 0.5. See the ensemble README for exact experiment design, results and limitations.
4. **GUI demonstration:** Run previously evaluated cases for regression/integration checks. This is not a new independent test or clinical validation.

## Quick start: GUI

Clone the repository and change into its root directory:

```bash
git clone https://github.com/wtwenting/multimodalLungDiseaseClassifier.git
cd multimodalLungDiseaseClassifier
```

**macOS/Linux:**

```bash
bash 1.Setup_and_Run.sh
```

**Windows Command Prompt:**

```bat
1.Setup_and_Run.cmd
```

The launchers set up a virtual environment, install dependencies from the top-level `requirements.txt`, and start `GUI/GUI_demo.py`. The application typically reports a local URL (often `http://127.0.0.1:7860`). Installation and model loading may depend on your operating system, Python version, available hardware and checkpoint files. The top-level requirements currently include a CUDA-oriented PyTorch index; non-CUDA platforms may require an adjusted PyTorch installation.

**Manual setup (alternative):**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python GUI/GUI_demo.py
```

For experiments, use the workstream-specific README instead of assuming the GUI setup alone reproduces training results.

## Demo data: what is actually included

The two demo packages use **flat per-patient folders**; the files are *not* nested inside `ct/` or `xray/` subfolders.

```text
Demo/
  Patient_Data_Raw/
    patient_01/
      scan.nii.gz
      image_01.png
      ...
    patient_04/
      image_01.png           # no raw CT in this repository revision
  Patient_Data_Preprocessed/
    patient_01/
      ct_v5_preprocessed.npz
      image_01_model_b_input.npy
      image_01_model_c_input.npy
      ...
    patient_04/
      ct_v5_preprocessed.npz # available despite missing raw CT
      image_01_model_b_input.npy
      image_01_model_c_input.npy
```

In the inspected `main` revision, raw CT NIfTI files are present for patients 01, 02, 03 and 05; **patient_04 has no raw CT**. Raw PNG X-rays are present for all five patients (13 images total). Preprocessed CT and X-ray arrays are present for all five. Therefore, raw CT + X-ray end-to-end regression is limited to the four patients with raw CT; the five-patient preprocessed-input demo remains available. The demo manifests may contain legacy `ct/` path strings: inspect the actual directory and relevant GUI code rather than assuming those strings are valid filesystem paths.

The cases were previously evaluated; do not use their recorded labels or predictions for tuning.

## Experiments and reproducibility

- [Cough experiment scripts, commands and archived summaries](cough/README.md)
- [CT + X-ray experiment mapping, commands and archived summaries](ensemble/README.md)
- [Archived ensemble experiment records](ensemble/experiment_records/)
- [Raw demo package notes](Demo/Patient_Data_Raw/README.md)
- [Preprocessed demo package notes](Demo/Patient_Data_Preprocessed/README.md)

The selected fusion evaluation reported 99 test patients. Its results and comparisons must be interpreted with the limitations in the ensemble README, including dataset-specific evaluation, uncertainty and patient-level (not verified same-session) pairing. Some original datasets, feature caches, split manifests and model artifacts are not bundled in this repository; full training reproduction requires obtaining the permitted data and matching artifacts.

## Continuous integration and branches

GitHub Actions runs the workflow in `.github/workflows/python-check.yml` for pull requests and pushes to `main`. Branches may contain changes not yet merged into `main`. In particular, the restored COUGHVID CLI, test fixtures and dependency declarations were available on the `cough` and `ensemble` branches during the repository review but absent from the inspected `main` revision. A successful local test or a passing feature-branch check does not establish that the current `main` checkout passes CI. Check the latest GitHub Actions run and the relevant pull request before relying on the status.

## Data use and limitations

Respect the original dataset terms and redistribution permissions (including BIMCV, COUGHVID and Coswara). Avoid committing identifiable patient data, unreviewed medical images, credentials or private local file paths. Cough-audio and CT/X-ray experiments are separate research contributions; the current selected frozen fusion classifier combines **CT and X-ray only**. No clinical diagnostic claim is made.
