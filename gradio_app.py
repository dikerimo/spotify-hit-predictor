from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from audio_features import extract_cover_features


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
CUSTOM_CSS = """
body, .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(161, 227, 188, 0.35), transparent 26%),
        radial-gradient(circle at top right, rgba(207, 239, 221, 0.45), transparent 30%),
        linear-gradient(180deg, #f4fbf6 0%, #edf7f0 48%, #e9f3ec 100%);
    color: #14251c;
    font-family: "Inter", "Segoe UI", sans-serif;
}

.gradio-container {
    max-width: 1180px !important;
    padding-top: 28px !important;
    padding-bottom: 40px !important;
}

#hero-card, #upload-card, #result-card, #meta-card {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(244, 251, 246, 0.94));
    border: 1px solid rgba(77, 142, 100, 0.12);
    border-radius: 28px;
    box-shadow: 0 18px 50px rgba(72, 109, 84, 0.10);
    backdrop-filter: blur(14px);
}

#hero-card {
    padding: 28px 30px 18px 30px;
    margin-bottom: 18px;
}

#upload-card, #result-card, #meta-card {
    padding: 18px;
}

.eyebrow {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(106, 189, 129, 0.10);
    border: 1px solid rgba(106, 189, 129, 0.18);
    color: #4c9362;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 18px;
}

.hero-title {
    margin: 14px 0 10px 0;
    font-size: 46px;
    line-height: 1.02;
    font-weight: 700;
    letter-spacing: -0.04em;
}

.hero-copy {
    max-width: 680px;
    color: #557264;
    font-size: 15px;
    line-height: 1.65;
}

.hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 12px;
    margin-top: 20px;
    max-width: 240px;
}

.hero-stat {
    padding: 14px 16px;
    border-radius: 20px;
    background: rgba(129, 217, 157, 0.08);
    border: 1px solid rgba(129, 217, 157, 0.14);
}

.hero-stat-label {
    color: #6f8879;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.hero-stat-value {
    margin-top: 8px;
    font-size: 20px;
    font-weight: 600;
    color: #183126;
}

.tree-sticker {
    width: 134px;
    height: 134px;
    border-radius: 30px;
    background: linear-gradient(180deg, #fbfffc 0%, #ecf8ef 100%);
    border: 1px solid rgba(112, 176, 130, 0.18);
    box-shadow: 0 14px 30px rgba(86, 127, 99, 0.12);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 74px;
    transform: rotate(-6deg);
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 8px;
}

.section-copy {
    color: #668273;
    font-size: 14px;
    line-height: 1.65;
    margin-bottom: 6px;
}

button.primary {
    background: linear-gradient(135deg, #80d49a 0%, #59b678 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    min-height: 52px !important;
}

button.primary:hover {
    filter: brightness(1.04);
}

.gr-button-secondary {
    border-radius: 16px !important;
}

.gradio-container .block {
    border: none !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .wrap,
.gradio-container .gr-box,
.gradio-container .gr-input,
.gradio-container .gr-form {
    background: rgba(255, 255, 255, 0.65) !important;
}

.gradio-container .gr-input, .gradio-container .gr-dropdown, .gradio-container .gr-audio {
    border-radius: 18px !important;
}

.result-chip-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 14px;
}

.result-chip {
    padding: 14px 16px;
    border-radius: 18px;
    background: rgba(129, 217, 157, 0.08);
    border: 1px solid rgba(129, 217, 157, 0.14);
}

.result-chip-label {
    color: #73907f;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.result-chip-value {
    margin-top: 8px;
    font-size: 22px;
    font-weight: 700;
    color: #183126;
}

@media (max-width: 900px) {
    .hero-top {
        flex-direction: column;
    }

    .hero-grid,
    .result-chip-row {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 34px;
    }
}
"""


def available_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(str(path) for path in MODELS_DIR.glob("*.joblib"))


def predict_song(audio_file: str, model_path: str) -> tuple[str, dict, dict]:
    if not audio_file:
        raise gr.Error("Upload an audio file first.")
    if not model_path:
        raise gr.Error("Select a model file first.")

    artifact = joblib.load(model_path)
    feature_names = artifact["feature_names"]
    model = artifact["model"]
    threshold = artifact["threshold"]

    extracted = extract_cover_features(audio_file)
    row = pd.DataFrame([{name: extracted.get(name) for name in feature_names}])

    probability = float(model.predict_proba(row)[0, 1])
    predicted_label = int(probability >= 0.5)
    verdict = "Stronger hit-aligned profile" if predicted_label == 1 else "Weaker hit-aligned profile"

    summary = {
        "audio_file": audio_file,
        "model_path": model_path,
        "training_popularity_threshold": threshold,
        "predicted_hit_probability": probability,
        "predicted_label": predicted_label,
        "verdict": verdict,
    }
    model_features = {name: extracted.get(name) for name in feature_names}
    analysis_features = {key: value for key, value in extracted.items() if key not in feature_names}

    explanation = (
        f"<div class='section-title'>{verdict}</div>"
        f"<div class='section-copy'>Use this as a general support signal when evaluating a song. "
        f"It reflects model alignment, not a guaranteed real-world outcome.</div>"
        f"<div class='result-chip-row'>"
        f"<div class='result-chip'><div class='result-chip-label'>Hit-Likeness</div><div class='result-chip-value'>{probability:.3f}</div></div>"
        f"<div class='result-chip'><div class='result-chip-label'>Model Decision</div><div class='result-chip-value'>{predicted_label}</div></div>"
        f"<div class='result-chip'><div class='result-chip-label'>Hit Threshold</div><div class='result-chip-value'>{threshold}</div></div>"
        f"</div>"
    )
    return explanation, summary, {"model_features": model_features, "analysis_features": analysis_features}


with gr.Blocks(title="Hit Detector", css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="hero-card"):
        gr.HTML(
            """
            <div class="hero-top">
                <div>
                    <div class="hero-title">Spotify Hit Predictor</div>
                    <div class="hero-copy">
                        Upload a song, run local audio analysis, and estimate how strongly its audio profile aligns with patterns learned from historically successful tracks.
                    </div>
                    <div class="hero-grid">
                        <div class="hero-stat">
                            <div class="hero-stat-label">Model Type</div>
                            <div class="hero-stat-value">Tuned Random Forest</div>
                        </div>
                    </div>
                </div>
                <div class="tree-sticker">🌳</div>
            </div>
            """
        )

    with gr.Row():
        with gr.Column(elem_id="upload-card", scale=1):
            gr.HTML("<div class='section-title'>Input</div><div class='section-copy'>Upload a song file and choose the trained model you want to use.</div>")
            audio_input = gr.Audio(type="filepath", label="Song File")
            model_dropdown = gr.Dropdown(
                choices=available_models(),
                value=(available_models()[0] if available_models() else None),
                label="Model File",
                allow_custom_value=True,
            )
            predict_button = gr.Button("Run Analysis", variant="primary")

    with gr.Row():
        with gr.Column(elem_id="result-card", scale=4):
            summary_output = gr.HTML(label="Summary")
        with gr.Column(elem_id="meta-card", scale=4):
            prediction_json = gr.JSON(label="Prediction")

    with gr.Row():
        with gr.Column(elem_id="meta-card"):
            feature_json = gr.JSON(label="Extracted Features")

    predict_button.click(
        fn=predict_song,
        inputs=[audio_input, model_dropdown],
        outputs=[summary_output, prediction_json, feature_json],
    )


if __name__ == "__main__":
    demo.launch()
