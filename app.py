"""
Deepfake Detection - Streamlit App
-------------------------------------
Full pipeline:
  1. Upload a face image or use demo
  2. CNN classifies: Real or Fake
  3. CNN-LSTM analyzes spatial patterns + attention map
  4. DeepSeek LLM explains everything

Run: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cnn_model import get_model as get_cnn
from models.cnn_lstm_model import get_model as get_cnn_lstm
from models.llm_assistant import DeepfakeAssistant
from data.data_loader import val_transform, inv_normalize, CLASS_NAMES, NUM_CLASSES


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
REGION_NAMES = [
    "forehead-L", "forehead", "forehead-R",
    "L-eye area", "nose bridge", "R-eye area", "R-temple",
    "L-cheek upper", "nose", "R-cheek upper", "R-cheek",
    "L-cheek", "mouth area", "R-cheek lower", "R-jaw",
    "L-jaw", "chin-L", "chin", "chin-R", "jaw-R",
]


def _get_region_name(idx: int) -> str:
    """Map attention index to approximate face region name."""
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


# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    st.title("🔍 Deepfake Detection & AI Analysis")
    st.markdown(
        "**Detect AI-generated fake faces using CNN + LSTM + DeepSeek LLM**"
    )
    st.divider()

    cnn_model, lstm_model = load_models()
    assistant = load_assistant()

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("Settings")
    data_source = st.sidebar.radio(
        "Image source",
        ["Upload your own image", "Use demo"],
    )
    api_key_input = st.sidebar.text_input(
        "DeepSeek API Key (optional)", type="password",
    )
    if api_key_input:
        assistant.set_api_key(api_key_input)

    # ── Step 1: Image Input ───────────────────────────────────────────────
    st.header("Step 1 - Face Image")

    image = None
    if data_source == "Upload your own image":
        uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", width=300)
    else:
        # Load a random image from test set if available
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
                st.image(image, caption=f"Demo image (Ground truth: {class_choice})", width=300)
        else:
            # Synthetic fallback
            np.random.seed(42)
            img_arr = np.random.randint(150, 220, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(img_arr)
            st.image(image, caption="Synthetic demo (no dataset found)", width=300)

    st.divider()

    # ── Step 2: AI Analysis ───────────────────────────────────────────────
    st.header("Step 2 - AI Analysis")

    if image is None:
        st.info("Please upload an image or select demo mode.")
        return

    if st.button("Analyse Image", type="primary"):
        with st.spinner("Running CNN analysis..."):
            cnn_result = predict_cnn(cnn_model, image)

        with st.spinner("Running CNN-LSTM spatial analysis..."):
            lstm_result = predict_cnn_lstm(lstm_model, image)

        st.session_state["cnn_result"] = cnn_result
        st.session_state["lstm_result"] = lstm_result
        assistant.set_detection_context(cnn_result, lstm_result)
        st.session_state["assistant"] = assistant
        st.session_state["chat_history"] = []

    if "cnn_result" in st.session_state and "lstm_result" in st.session_state:
        cnn_result = st.session_state["cnn_result"]
        lstm_result = st.session_state["lstm_result"]

        # Ensemble verdict
        avg_fake = (cnn_result["prob_fake"] + lstm_result["prob_fake"]) / 2
        verdict = "FAKE" if avg_fake > 0.5 else "REAL"
        verdict_color = "🔴" if verdict == "FAKE" else "🟢"

        st.subheader(f"{verdict_color} Overall Verdict: {verdict} (Combined confidence: {max(avg_fake, 1-avg_fake)*100:.1f}%)")
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            is_fake = cnn_result["class_id"] == 1
            icon = "🔴" if is_fake else "🟢"
            st.metric(
                f"{icon} CNN Prediction",
                cnn_result["prediction"],
                f"Confidence: {cnn_result['confidence'] * 100:.1f}%",
            )
            st.progress(cnn_result["prob_fake"])
            st.caption(f"Fake probability: {cnn_result['prob_fake'] * 100:.1f}%")

        with col2:
            is_fake_lstm = lstm_result["prediction"] == "Fake"
            icon2 = "🔴" if is_fake_lstm else "🟢"
            st.metric(
                f"{icon2} CNN-LSTM Prediction",
                lstm_result["prediction"],
                f"Confidence: {lstm_result['confidence'] * 100:.1f}%",
            )
            st.progress(lstm_result["prob_fake"])
            st.caption(f"Most suspicious: {lstm_result['suspicious_region']}")

        # Attention map visualization
        st.subheader("Spatial Attention Map - Where did the AI look?")
        attn_map = lstm_result["attention_map"]

        col3, col4 = st.columns(2)
        with col3:
            st.image(image, caption="Original Image", width=300)

        with col4:
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(attn_map, cmap="hot", interpolation="bilinear")
            ax.set_title("LSTM Attention (bright = more focus)")
            ax.set_xlabel("← Left face | Right face →")
            ax.set_ylabel("← Forehead | Chin →")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig)
            plt.close(fig)

        # Overlay
        st.subheader("Attention Overlay on Face")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        img_resized = image.resize((224, 224))
        ax2.imshow(img_resized)
        # Upsample attention map to image size
        attn_upsampled = np.array(Image.fromarray(
            (attn_map * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)) / 255.0
        ax2.imshow(attn_upsampled, cmap="jet", alpha=0.4)
        ax2.set_title("Face regions the AI focused on most")
        ax2.axis("off")
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    # ── Step 3: LLM Chat ─────────────────────────────────────────────────
    st.header("Step 3 - Ask the AI Assistant")
    st.markdown("Ask questions about the detection results. Powered by **DeepSeek**.")

    if "assistant" not in st.session_state:
        st.info("Click **Analyse Image** first to enable the assistant.")
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
