"""
Deepfake Detection - Streamlit App
-------------------------------------
Full pipeline:
  1. Upload a face image OR video
  2. CNN classifies: Real or Fake
  3. CNN-LSTM analyzes spatial patterns + attention map
  4. For video: frame-by-frame timeline analysis
  5. DeepSeek LLM explains everything

Run: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cnn_model import get_model as get_cnn
from models.cnn_lstm_model import get_model as get_cnn_lstm
from models.llm_assistant import DeepfakeAssistant
from data.data_loader import (
    val_transform, inv_normalize, CLASS_NAMES, NUM_CLASSES,
    extract_frames_from_video, get_video_info,
)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Deepfake Detector AI", page_icon="🔍", layout="wide")


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cnn = get_cnn(num_classes=NUM_CLASSES)
    cnn_lstm = get_cnn_lstm(num_classes=NUM_CLASSES)
    cnn.eval()
    cnn_lstm.eval()

    if os.path.exists("checkpoints/cnn_best.pt"):
        cnn.load_state_dict(torch.load("checkpoints/cnn_best.pt", map_location="cpu"))
        st.sidebar.success("CNN weights loaded")
    else:
        st.sidebar.warning("CNN using random weights - train first!")

    if os.path.exists("checkpoints/cnn_lstm_best.pt"):
        cnn_lstm.load_state_dict(torch.load("checkpoints/cnn_lstm_best.pt", map_location="cpu"))
        st.sidebar.success("CNN-LSTM weights loaded")
    else:
        st.sidebar.warning("CNN-LSTM using random weights - train first!")

    return cnn, cnn_lstm


@st.cache_resource
def load_assistant():
    return DeepfakeAssistant()


# ── Prediction helpers ────────────────────────────────────────────────────────
def _get_region_name(idx: int) -> str:
    regions_7x7 = [
        "top-left forehead", "upper forehead", "mid forehead",
        "upper right forehead", "right forehead", "right temple", "right hairline",
        "left eye area", "left eyebrow", "between eyes", "right eyebrow",
        "right eye area", "right temple", "right side",
        "left cheek (upper)", "left eye", "nose bridge", "right eye",
        "right cheek (upper)", "right side", "right ear area",
        "left cheek", "left nostril", "nose tip", "right nostril",
        "right cheek", "right jaw upper", "right jaw",
        "left jaw", "left mouth", "lips", "right mouth",
        "right jaw lower", "right chin area", "right neck",
        "left jaw lower", "left chin", "chin center", "right chin",
        "chin right", "lower right", "right neck lower",
        "left neck", "lower left", "lower center left", "lower center",
        "lower center right", "lower right", "lower right corner",
    ]
    if idx < len(regions_7x7):
        return regions_7x7[idx]
    return f"region {idx}"


def predict_cnn(model, image: Image.Image) -> dict:
    tensor = val_transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    class_id = probs.argmax().item()
    return {
        "prediction": CLASS_NAMES[class_id],
        "confidence": probs[class_id].item(),
        "prob_real": probs[0].item(),
        "prob_fake": probs[1].item(),
        "class_id": class_id,
    }


def predict_cnn_lstm(model, image: Image.Image) -> dict:
    tensor = val_transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        attn_map = model.get_attention_map(tensor)[0].numpy()
    class_id = probs.argmax().item()
    max_attn_idx = attn_map.flatten().argmax()
    return {
        "prediction": CLASS_NAMES[class_id],
        "confidence": probs[class_id].item(),
        "prob_real": probs[0].item(),
        "prob_fake": probs[1].item(),
        "attention_map": attn_map,
        "suspicious_region": _get_region_name(max_attn_idx),
        "attention_description": f"Highest focus on {_get_region_name(max_attn_idx)}",
    }


def analyse_single_image(cnn_model, lstm_model, image):
    """Run both models on a single image and return results."""
    cnn_result = predict_cnn(cnn_model, image)
    lstm_result = predict_cnn_lstm(lstm_model, image)
    return cnn_result, lstm_result


def analyse_video_frames(cnn_model, lstm_model, frames, progress_bar=None):
    """Run both models on each frame. Returns per-frame results."""
    cnn_results = []
    lstm_results = []

    for i, frame in enumerate(frames):
        cnn_r = predict_cnn(cnn_model, frame)
        lstm_r = predict_cnn_lstm(lstm_model, frame)
        cnn_results.append(cnn_r)
        lstm_results.append(lstm_r)

        if progress_bar:
            progress_bar.progress((i + 1) / len(frames))

    return cnn_results, lstm_results


# ── Display helpers ───────────────────────────────────────────────────────────
def show_image_results(cnn_result, lstm_result, image):
    """Display analysis results for a single image."""
    # Ensemble verdict
    avg_fake = (cnn_result["prob_fake"] + lstm_result["prob_fake"]) / 2
    verdict = "FAKE" if avg_fake > 0.5 else "REAL"
    verdict_color = "🔴" if verdict == "FAKE" else "🟢"

    st.subheader(f"{verdict_color} Verdict: {verdict} (Combined confidence: {max(avg_fake, 1 - avg_fake) * 100:.1f}%)")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        is_fake = cnn_result["class_id"] == 1
        icon = "🔴" if is_fake else "🟢"
        st.metric(f"{icon} CNN Prediction", cnn_result["prediction"],
                  f"Confidence: {cnn_result['confidence'] * 100:.1f}%")
        st.progress(cnn_result["prob_fake"])
        st.caption(f"Fake probability: {cnn_result['prob_fake'] * 100:.1f}%")

    with col2:
        is_fake_lstm = lstm_result["prediction"] == "Fake"
        icon2 = "🔴" if is_fake_lstm else "🟢"
        st.metric(f"{icon2} CNN-LSTM Prediction", lstm_result["prediction"],
                  f"Confidence: {lstm_result['confidence'] * 100:.1f}%")
        st.progress(lstm_result["prob_fake"])
        st.caption(f"Most suspicious: {lstm_result['suspicious_region']}")

    # Attention map
    st.subheader("Spatial Attention Map")
    attn_map = lstm_result["attention_map"]

    col3, col4 = st.columns(2)
    with col3:
        st.image(image, caption="Original Image", width=300)
    with col4:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(attn_map, cmap="hot", interpolation="bilinear")
        ax.set_title("LSTM Attention (bright = more focus)")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
        plt.close(fig)

    # Overlay
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    img_resized = image.resize((224, 224))
    ax2.imshow(img_resized)
    attn_upsampled = np.array(Image.fromarray(
        (attn_map * 255).astype(np.uint8)
    ).resize((224, 224), Image.BILINEAR)) / 255.0
    ax2.imshow(attn_upsampled, cmap="jet", alpha=0.4)
    ax2.set_title("Attention Overlay")
    ax2.axis("off")
    st.pyplot(fig2)
    plt.close(fig2)


def show_video_results(cnn_results, lstm_results, frames, fps):
    """Display analysis results for a video."""
    n_frames = len(frames)

    # Per-frame fake probabilities
    cnn_fake_probs = [r["prob_fake"] for r in cnn_results]
    lstm_fake_probs = [r["prob_fake"] for r in lstm_results]
    ensemble_probs = [(c + l) / 2 for c, l in zip(cnn_fake_probs, lstm_fake_probs)]

    # Overall verdict
    avg_fake = np.mean(ensemble_probs)
    fake_frame_count = sum(1 for p in ensemble_probs if p > 0.5)
    fake_percentage = fake_frame_count / n_frames * 100

    verdict = "FAKE" if avg_fake > 0.5 else "REAL"
    verdict_color = "🔴" if verdict == "FAKE" else "🟢"

    st.subheader(f"{verdict_color} Video Verdict: {verdict}")
    st.markdown(
        f"- **Average fake probability**: {avg_fake * 100:.1f}%\n"
        f"- **Fake frames**: {fake_frame_count}/{n_frames} ({fake_percentage:.0f}%)\n"
        f"- **Frames analysed**: {n_frames}"
    )
    st.divider()

    # Timeline chart
    st.subheader("Frame-by-Frame Analysis Timeline")
    timestamps = [i / fps if fps > 0 else i for i in range(n_frames)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timestamps, cnn_fake_probs, label="CNN", color="#457b9d", alpha=0.7, linewidth=1.5)
    ax.plot(timestamps, lstm_fake_probs, label="CNN-LSTM", color="#e63946", alpha=0.7, linewidth=1.5)
    ax.plot(timestamps, ensemble_probs, label="Combined", color="#2d6a4f", linewidth=2.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold (0.5)")
    ax.fill_between(timestamps, ensemble_probs, 0.5,
                    where=[p > 0.5 for p in ensemble_probs],
                    color="#e63946", alpha=0.15, label="Fake region")
    ax.fill_between(timestamps, ensemble_probs, 0.5,
                    where=[p <= 0.5 for p in ensemble_probs],
                    color="#2a9d8f", alpha=0.15, label="Real region")
    ax.set_xlabel("Time (seconds)" if fps > 0 else "Frame")
    ax.set_ylabel("Fake Probability")
    ax.set_title("Deepfake Probability Over Time")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Most suspicious and most real frames
    st.subheader("Key Frames")
    most_fake_idx = np.argmax(ensemble_probs)
    most_real_idx = np.argmin(ensemble_probs)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Most suspicious frame** (#{most_fake_idx + 1}, "
                    f"fake prob: {ensemble_probs[most_fake_idx] * 100:.1f}%)")
        st.image(frames[most_fake_idx], width=300)

        # Show attention map for most suspicious frame
        attn_map = lstm_results[most_fake_idx]["attention_map"]
        fig3, ax3 = plt.subplots(figsize=(3, 3))
        img_resized = frames[most_fake_idx].resize((224, 224))
        ax3.imshow(img_resized)
        attn_up = np.array(Image.fromarray(
            (attn_map * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)) / 255.0
        ax3.imshow(attn_up, cmap="jet", alpha=0.4)
        ax3.set_title(f"Attention - suspicious: {lstm_results[most_fake_idx]['suspicious_region']}")
        ax3.axis("off")
        st.pyplot(fig3)
        plt.close(fig3)

    with col2:
        st.markdown(f"**Most authentic frame** (#{most_real_idx + 1}, "
                    f"fake prob: {ensemble_probs[most_real_idx] * 100:.1f}%)")
        st.image(frames[most_real_idx], width=300)

        attn_map2 = lstm_results[most_real_idx]["attention_map"]
        fig4, ax4 = plt.subplots(figsize=(3, 3))
        img_resized2 = frames[most_real_idx].resize((224, 224))
        ax4.imshow(img_resized2)
        attn_up2 = np.array(Image.fromarray(
            (attn_map2 * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)) / 255.0
        ax4.imshow(attn_up2, cmap="jet", alpha=0.4)
        ax4.set_title("Attention - authentic frame")
        ax4.axis("off")
        st.pyplot(fig4)
        plt.close(fig4)

    # Frame browser
    st.subheader("Browse All Frames")
    frame_idx = st.slider("Select frame", 1, n_frames, 1) - 1
    bcol1, bcol2, bcol3 = st.columns([2, 2, 1])
    with bcol1:
        st.image(frames[frame_idx], caption=f"Frame {frame_idx + 1}", width=250)
    with bcol2:
        attn_m = lstm_results[frame_idx]["attention_map"]
        fig5, ax5 = plt.subplots(figsize=(3, 3))
        ax5.imshow(attn_m, cmap="hot", interpolation="bilinear")
        ax5.set_title("Attention Map")
        ax5.axis("off")
        st.pyplot(fig5)
        plt.close(fig5)
    with bcol3:
        fr_verdict = "Fake" if ensemble_probs[frame_idx] > 0.5 else "Real"
        fr_icon = "🔴" if fr_verdict == "Fake" else "🟢"
        st.metric(f"{fr_icon} Frame {frame_idx + 1}", fr_verdict,
                  f"{ensemble_probs[frame_idx] * 100:.1f}% fake")
        st.caption(f"CNN: {cnn_results[frame_idx]['prob_fake'] * 100:.1f}%")
        st.caption(f"LSTM: {lstm_results[frame_idx]['prob_fake'] * 100:.1f}%")


# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    st.title("🔍 Deepfake Detection & AI Analysis")
    st.markdown(
        "**Detect AI-generated fake faces in images and videos using CNN + LSTM + DeepSeek LLM**"
    )
    st.divider()

    cnn_model, lstm_model = load_models()
    assistant = load_assistant()

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("Settings")

    input_type = st.sidebar.radio("Input type", ["Image", "Video"])

    api_key_input = st.sidebar.text_input(
        "DeepSeek API Key (optional)", type="password",
    )
    if api_key_input:
        assistant.set_api_key(api_key_input)

    if input_type == "Video":
        max_frames = st.sidebar.slider("Max frames to analyse", 5, 60, 20)

    # ── Step 1: Input ─────────────────────────────────────────────────────
    st.header(f"Step 1 - Upload {input_type}")

    image = None
    frames = None
    fps = 30

    if input_type == "Image":
        source = st.sidebar.radio("Source", ["Upload", "Demo"])

        if source == "Upload":
            uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Uploaded image", width=300)
        else:
            test_dir = "data/Dataset/Test"
            if os.path.isdir(test_dir):
                seed = st.sidebar.slider("Demo seed", 0, 99, 42)
                np.random.seed(seed)
                class_choice = np.random.choice(["Real", "Fake"])
                class_dir = os.path.join(test_dir, class_choice)
                files = [f for f in os.listdir(class_dir) if f.endswith((".jpg", ".png"))]
                if files:
                    chosen = files[seed % len(files)]
                    image = Image.open(os.path.join(class_dir, chosen)).convert("RGB")
                    st.image(image, caption=f"Demo (Ground truth: {class_choice})", width=300)
            else:
                st.warning("Test dataset not found. Upload an image instead.")

    else:  # Video
        uploaded_video = st.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"],
        )
        if uploaded_video:
            # Save to temp file (cv2 needs a file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            # Show video
            st.video(uploaded_video)

            # Get video info
            try:
                info = get_video_info(tmp_path)
                st.markdown(
                    f"**Video info**: {info['width']}x{info['height']}, "
                    f"{info['fps']:.1f} FPS, {info['duration_sec']:.1f}s, "
                    f"{info['total_frames']} total frames"
                )

                # Extract frames
                with st.spinner(f"Extracting {max_frames} frames..."):
                    frames, fps, total = extract_frames_from_video(
                        tmp_path, max_frames=max_frames
                    )
                st.success(f"Extracted {len(frames)} frames from video")

                # Show sample frames
                with st.expander("Preview extracted frames"):
                    preview_cols = st.columns(min(6, len(frames)))
                    for i, col in enumerate(preview_cols):
                        idx = i * (len(frames) // len(preview_cols))
                        if idx < len(frames):
                            col.image(frames[idx], caption=f"#{idx + 1}", width=100)
            except Exception as e:
                st.error(f"Error processing video: {e}")
                frames = None
            finally:
                os.unlink(tmp_path)

    st.divider()

    # ── Step 2: Analysis ──────────────────────────────────────────────────
    st.header("Step 2 - AI Analysis")

    if input_type == "Image":
        if image is None:
            st.info("Please upload an image or select demo mode.")
            return

        if st.button("Analyse Image", type="primary"):
            with st.spinner("Running analysis..."):
                cnn_result, lstm_result = analyse_single_image(cnn_model, lstm_model, image)

            st.session_state["cnn_result"] = cnn_result
            st.session_state["lstm_result"] = lstm_result
            st.session_state["mode"] = "image"
            assistant.set_detection_context(cnn_result, lstm_result)
            st.session_state["assistant"] = assistant
            st.session_state["chat_history"] = []

        if st.session_state.get("mode") == "image" and "cnn_result" in st.session_state:
            show_image_results(
                st.session_state["cnn_result"],
                st.session_state["lstm_result"],
                image,
            )

    else:  # Video
        if frames is None:
            st.info("Please upload a video to proceed.")
            return

        if st.button("Analyse Video", type="primary"):
            progress = st.progress(0)
            st.write(f"Analysing {len(frames)} frames...")

            cnn_results, lstm_results = analyse_video_frames(
                cnn_model, lstm_model, frames, progress_bar=progress
            )

            st.session_state["video_cnn_results"] = cnn_results
            st.session_state["video_lstm_results"] = lstm_results
            st.session_state["video_frames"] = frames
            st.session_state["video_fps"] = fps
            st.session_state["mode"] = "video"

            # Set context for LLM using aggregated results
            avg_cnn_fake = np.mean([r["prob_fake"] for r in cnn_results])
            avg_lstm_fake = np.mean([r["prob_fake"] for r in lstm_results])
            fake_count = sum(1 for c, l in zip(cnn_results, lstm_results)
                           if (c["prob_fake"] + l["prob_fake"]) / 2 > 0.5)

            summary_cnn = {
                "prediction": "Fake" if avg_cnn_fake > 0.5 else "Real",
                "confidence": max(avg_cnn_fake, 1 - avg_cnn_fake),
                "prob_real": 1 - avg_cnn_fake,
                "prob_fake": avg_cnn_fake,
            }
            summary_lstm = {
                "prediction": "Fake" if avg_lstm_fake > 0.5 else "Real",
                "confidence": max(avg_lstm_fake, 1 - avg_lstm_fake),
                "prob_real": 1 - avg_lstm_fake,
                "prob_fake": avg_lstm_fake,
                "suspicious_region": lstm_results[np.argmax(
                    [r["prob_fake"] for r in lstm_results]
                )]["suspicious_region"],
                "attention_description": (
                    f"{fake_count}/{len(frames)} frames flagged as fake"
                ),
            }
            assistant.set_detection_context(summary_cnn, summary_lstm)
            st.session_state["assistant"] = assistant
            st.session_state["chat_history"] = []

        if (st.session_state.get("mode") == "video"
                and "video_cnn_results" in st.session_state):
            show_video_results(
                st.session_state["video_cnn_results"],
                st.session_state["video_lstm_results"],
                st.session_state["video_frames"],
                st.session_state["video_fps"],
            )

    st.divider()

    # ── Step 3: LLM Chat ─────────────────────────────────────────────────
    st.header("Step 3 - Ask the AI Assistant")
    st.markdown("Ask questions about the detection results. Powered by **DeepSeek**.")

    if "assistant" not in st.session_state:
        st.info("Click **Analyse** first to enable the assistant.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about the deepfake analysis..."):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = st.session_state["assistant"].chat(prompt)
                    except Exception as e:
                        reply = (
                            f"Could not connect to DeepSeek API: {e}\n\n"
                            "Please set your DEEPSEEK_API_KEY environment variable."
                        )
                st.markdown(reply)
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})

    st.divider()
    st.caption(
        "No AI detector is 100% accurate. Deepfake technology is constantly evolving. "
        "This tool is for educational and research purposes only."
    )


if __name__ == "__main__":
    main()
