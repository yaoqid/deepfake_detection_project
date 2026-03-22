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
import cv2
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


# ── Face Detection ────────────────────────────────────────────────────────────
@st.cache_resource
def load_face_detector():
    """Load OpenCV's DNN face detector (more accurate than Haar cascades)."""
    # Use OpenCV's built-in Haar cascade as fallback (always available)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    return detector


def detect_and_crop_face(image: Image.Image, detector, margin: float = 0.3) -> Image.Image:
    """
    Detect the largest face in the image and crop it.
    If no face is found, returns the original image (center-cropped).
    Uses asymmetric margin: extra padding below for mouth/chin area.

    Args:
        image: PIL Image
        detector: OpenCV cascade classifier
        margin: extra margin around face (0.3 = 30% padding)

    Returns:
        Cropped face as PIL Image
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Try multiple detector settings for better face detection
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    if len(faces) == 0:
        # Retry with more lenient settings
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )

    if len(faces) == 0:
        # No face found - do a center crop (assume face is centered)
        h, w = img_array.shape[:2]
        size = min(h, w)
        top = (h - size) // 2
        left = (w - size) // 2
        return image.crop((left, top, left + size, top + size))

    # Pick the largest face
    areas = [fw * fh for (fx, fy, fw, fh) in faces]
    largest = faces[np.argmax(areas)]
    x, y, w, h = largest

    # Asymmetric margin: more padding below (mouth/chin) and sides
    img_h, img_w = img_array.shape[:2]
    margin_x = int(w * margin)
    margin_top = int(h * margin)
    margin_bottom = int(h * (margin + 0.25))  # 55% below for mouth/chin

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_top)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_bottom)

    return image.crop((x1, y1, x2, y2))


def preprocess_for_model(image: Image.Image, detector) -> Image.Image:
    """Detect face, crop it, and return the cropped face image."""
    return detect_and_crop_face(image, detector)


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


def artifact_analysis(image: Image.Image) -> dict:
    """
    Multi-signal artifact analysis - detects manipulation without ML.

    Three complementary signals:
    1. Noise analysis: manipulated regions have different noise patterns
    2. Blur inconsistency: generated/pasted areas have different sharpness
    3. ELA: compression level differences between regions

    Returns dict with fake probability, heatmap, and suspicious region info.
    """
    import io

    img_resized = image.resize((224, 224))
    img_arr = np.array(img_resized).astype(np.float32)
    gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY).astype(np.float32)

    patch_h, patch_w = 224 // 7, 224 // 7

    # ── Signal 1: Noise variance per region ─────────────────────────────
    # Manipulated areas have different noise characteristics
    # Use high-pass filter to isolate noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.abs(gray - blurred)

    noise_scores = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            patch = noise[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            noise_scores[i, j] = patch.std()

    # High deviation from median = suspicious (different noise pattern)
    noise_median = np.median(noise_scores)
    noise_deviation = np.abs(noise_scores - noise_median)
    noise_deviation = noise_deviation / (noise_deviation.max() + 1e-8)

    # ── Signal 2: Blur/sharpness inconsistency (Laplacian) ──────────────
    # Deepfake boundaries often have blur mismatches
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    lap_abs = np.abs(laplacian)

    sharpness_scores = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            patch = lap_abs[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            sharpness_scores[i, j] = patch.var()

    # Regions that differ from surroundings are suspicious
    sharp_median = np.median(sharpness_scores)
    sharp_deviation = np.abs(sharpness_scores - sharp_median)
    sharp_deviation = sharp_deviation / (sharp_deviation.max() + 1e-8)

    # ── Signal 3: ELA (Error Level Analysis) ────────────────────────────
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")
    resaved_arr = np.array(resaved.resize((224, 224))).astype(np.float32)
    ela_diff = np.abs(img_arr - resaved_arr).mean(axis=2)
    ela_scaled = ela_diff / (ela_diff.max() + 1e-8)

    ela_scores = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            patch = ela_scaled[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            ela_scores[i, j] = patch.mean()

    ela_median = np.median(ela_scores)
    ela_deviation = np.abs(ela_scores - ela_median)
    ela_deviation = ela_deviation / (ela_deviation.max() + 1e-8)

    # ── Combine all signals ─────────────────────────────────────────────
    # Weight: noise and blur are more reliable than ELA for non-JPEG images
    combined_scores = (noise_deviation * 0.4 + sharp_deviation * 0.4 + ela_deviation * 0.2)
    combined_scores = combined_scores / (combined_scores.max() + 1e-8)

    # Generate pixel-level heatmap for display
    heatmap = np.zeros((224, 224), dtype=np.float32)
    for i in range(7):
        for j in range(7):
            heatmap[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = combined_scores[i, j]

    # ── Compute fake probability ────────────────────────────────────────
    # Key insight: real faces have CONSISTENT noise/blur across the face
    # Fakes have INCONSISTENT patterns (generated region vs original)
    overall_variance = np.std(combined_scores)
    # Check if there are clear outlier regions (manipulation hotspots)
    top_3 = np.sort(combined_scores.flatten())[-3:]
    bottom_3 = np.sort(combined_scores.flatten())[:3]
    contrast = top_3.mean() - bottom_3.mean()

    # Fake probability: high variance + high contrast = likely manipulated
    fake_prob = min(1.0, overall_variance * 3.0 + contrast * 0.8)

    max_idx = combined_scores.flatten().argmax()

    return {
        "ela_map": heatmap,  # (224, 224) for display
        "region_scores": combined_scores,  # (7, 7) grid
        "ela_fake_prob": fake_prob,
        "suspicious_region": _get_region_name(max_idx),
        "noise_var": np.std(noise_scores),
        "blur_var": np.std(sharpness_scores),
    }


def _tta_augmentations(image: Image.Image) -> list:
    """Generate test-time augmentation variants of an image."""
    from PIL import ImageEnhance
    augmented = [image]
    # Horizontal flip
    augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    # Slight brightness changes
    enhancer = ImageEnhance.Brightness(image)
    augmented.append(enhancer.enhance(0.9))
    augmented.append(enhancer.enhance(1.1))
    # Slight contrast change
    enhancer_c = ImageEnhance.Contrast(image)
    augmented.append(enhancer_c.enhance(0.9))
    return augmented


def predict_cnn(model, image: Image.Image, use_tta: bool = True) -> dict:
    if use_tta:
        all_probs = []
        for aug_img in _tta_augmentations(image):
            tensor = val_transform(aug_img).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                all_probs.append(probs)
        probs = torch.stack(all_probs).mean(dim=0)
    else:
        tensor = val_transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

    # Grad-CAM: which regions push toward "Fake"
    tensor = val_transform(image).unsqueeze(0)
    gradcam = model.get_gradcam(tensor, target_class=1)[0].numpy()  # (7, 7)

    class_id = probs.argmax().item()
    return {
        "prediction": CLASS_NAMES[class_id],
        "confidence": probs[class_id].item(),
        "prob_real": probs[0].item(),
        "prob_fake": probs[1].item(),
        "class_id": class_id,
        "gradcam": gradcam,
    }


def predict_cnn_lstm(model, image: Image.Image, use_tta: bool = True) -> dict:
    # Grad-CAM: gradient-weighted attention showing which patches drive "Fake"
    tensor = val_transform(image).unsqueeze(0)
    gradcam_map = model.get_fake_gradcam(tensor)[0].numpy()  # (7, 7)

    # Also get raw attention for comparison
    with torch.no_grad():
        raw_attn = model.get_attention_map(tensor)[0].numpy()

    if use_tta:
        all_probs = []
        for aug_img in _tta_augmentations(image):
            t = val_transform(aug_img).unsqueeze(0)
            with torch.no_grad():
                logits = model(t)
                probs = torch.softmax(logits, dim=1)[0]
                all_probs.append(probs)
        probs = torch.stack(all_probs).mean(dim=0)
    else:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

    class_id = probs.argmax().item()
    max_attn_idx = gradcam_map.flatten().argmax()
    return {
        "prediction": CLASS_NAMES[class_id],
        "confidence": probs[class_id].item(),
        "prob_real": probs[0].item(),
        "prob_fake": probs[1].item(),
        "attention_map": gradcam_map,
        "raw_attention": raw_attn,
        "suspicious_region": _get_region_name(max_attn_idx),
        "attention_description": f"Highest focus on {_get_region_name(max_attn_idx)}",
    }


def analyse_single_image(cnn_model, lstm_model, image, detector):
    """Detect face, crop it, then run both models + ELA."""
    face_crop = preprocess_for_model(image, detector)
    cnn_result = predict_cnn(cnn_model, face_crop)
    lstm_result = predict_cnn_lstm(lstm_model, face_crop)
    ela_result = artifact_analysis(face_crop)
    return cnn_result, lstm_result, ela_result, face_crop


def analyse_video_frames(cnn_model, lstm_model, frames, detector, progress_bar=None):
    """Detect face in each frame, crop, then run both models + ELA."""
    cnn_results = []
    lstm_results = []
    ela_results = []
    face_crops = []

    for i, frame in enumerate(frames):
        face_crop = preprocess_for_model(frame, detector)
        face_crops.append(face_crop)
        cnn_r = predict_cnn(cnn_model, face_crop)
        lstm_r = predict_cnn_lstm(lstm_model, face_crop)
        ela_r = artifact_analysis(face_crop)
        cnn_results.append(cnn_r)
        lstm_results.append(lstm_r)
        ela_results.append(ela_r)

        if progress_bar:
            progress_bar.progress((i + 1) / len(frames))

    return cnn_results, lstm_results, ela_results, face_crops


# ── Display helpers ───────────────────────────────────────────────────────────
def show_image_results(cnn_result, lstm_result, image, threshold=0.45, ela_result=None):
    """Display analysis results for a single image."""
    # Ensemble verdict: CNN + LSTM + ELA (3 signals)
    ela_fake = ela_result["ela_fake_prob"] if ela_result else 0.0
    avg_fake = (cnn_result["prob_fake"] + lstm_result["prob_fake"] + ela_fake) / 3
    verdict = "FAKE" if avg_fake > threshold else "REAL"
    verdict_color = "🔴" if verdict == "FAKE" else "🟢"

    st.subheader(f"{verdict_color} Verdict: {verdict} (Combined confidence: {max(avg_fake, 1 - avg_fake) * 100:.1f}%)")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        is_fake = cnn_result["class_id"] == 1
        icon = "🔴" if is_fake else "🟢"
        st.metric(f"{icon} CNN", cnn_result["prediction"],
                  f"Confidence: {cnn_result['confidence'] * 100:.1f}%")
        st.progress(cnn_result["prob_fake"])
        st.caption(f"Fake probability: {cnn_result['prob_fake'] * 100:.1f}%")

    with col2:
        is_fake_lstm = lstm_result["prediction"] == "Fake"
        icon2 = "🔴" if is_fake_lstm else "🟢"
        st.metric(f"{icon2} CNN-LSTM", lstm_result["prediction"],
                  f"Confidence: {lstm_result['confidence'] * 100:.1f}%")
        st.progress(lstm_result["prob_fake"])
        st.caption(f"Most suspicious: {lstm_result['suspicious_region']}")

    with col3:
        if ela_result:
            is_fake_ela = ela_fake > threshold
            icon3 = "🔴" if is_fake_ela else "🟢"
            ela_verdict = "Fake" if is_fake_ela else "Real"
            st.metric(f"{icon3} Artifact Scan", ela_verdict,
                      f"Score: {ela_fake * 100:.1f}%")
            st.progress(min(ela_fake, 1.0))
            st.caption(f"Hotspot: {ela_result['suspicious_region']}")

    # Heatmaps: show WHERE the models see fake artifacts
    st.subheader("Where the models see fake artifacts")
    st.caption("Bright/red areas = suspected manipulation. ELA detects compression inconsistencies without ML.")

    img_resized = image.resize((224, 224))
    lstm_map = lstm_result["attention_map"]
    cnn_map = cnn_result.get("gradcam")
    ela_map = ela_result["ela_map"] if ela_result else None

    # Show CNN Grad-CAM + LSTM Grad-CAM + ELA side by side
    heatmap_cols = st.columns(4)
    with heatmap_cols[0]:
        st.image(image, caption="Face crop (model input)", width=200)
    with heatmap_cols[1]:
        if cnn_map is not None:
            fig_cnn, ax_cnn = plt.subplots(figsize=(4, 4))
            ax_cnn.imshow(img_resized)
            cnn_up = np.array(Image.fromarray(
                (cnn_map * 255).astype(np.uint8)
            ).resize((224, 224), Image.BILINEAR)) / 255.0
            ax_cnn.imshow(cnn_up, cmap="jet", alpha=0.45)
            ax_cnn.set_title("CNN Grad-CAM")
            ax_cnn.axis("off")
            st.pyplot(fig_cnn)
            plt.close(fig_cnn)
    with heatmap_cols[2]:
        fig_lstm, ax_lstm = plt.subplots(figsize=(4, 4))
        ax_lstm.imshow(img_resized)
        lstm_up = np.array(Image.fromarray(
            (lstm_map * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)) / 255.0
        ax_lstm.imshow(lstm_up, cmap="jet", alpha=0.45)
        ax_lstm.set_title(f"LSTM Grad-CAM\n({lstm_result['suspicious_region']})")
        ax_lstm.axis("off")
        st.pyplot(fig_lstm)
        plt.close(fig_lstm)
    with heatmap_cols[3]:
        if ela_map is not None:
            fig_ela, ax_ela = plt.subplots(figsize=(4, 4))
            ax_ela.imshow(img_resized)
            ax_ela.imshow(ela_map, cmap="jet", alpha=0.5)
            ax_ela.set_title(f"Artifact Analysis\n({ela_result['suspicious_region']})")
            ax_ela.axis("off")
            st.pyplot(fig_ela)
            plt.close(fig_ela)

    # Combined overlay (all 3 signals)
    n_maps = 0
    combined_map = np.zeros((7, 7), dtype=np.float32)
    if cnn_map is not None:
        combined_map += cnn_map
        n_maps += 1
    combined_map += lstm_map
    n_maps += 1
    if ela_result:
        combined_map += ela_result["region_scores"]
        n_maps += 1
    combined_map /= n_maps

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(img_resized)
    combined_up = np.array(Image.fromarray(
        (combined_map * 255).astype(np.uint8)
    ).resize((224, 224), Image.BILINEAR)) / 255.0
    ax2.imshow(combined_up, cmap="jet", alpha=0.45)
    ax2.set_title("Combined Heatmap (CNN + LSTM + ELA)")
    ax2.axis("off")
    st.pyplot(fig2)
    plt.close(fig2)


def show_video_results(cnn_results, lstm_results, frames, fps, threshold=0.45, ela_results=None):
    """Display analysis results for a video."""
    n_frames = len(frames)

    # Per-frame fake probabilities (3-way ensemble if ELA available)
    cnn_fake_probs = [r["prob_fake"] for r in cnn_results]
    lstm_fake_probs = [r["prob_fake"] for r in lstm_results]
    if ela_results:
        ela_fake_probs = [r["ela_fake_prob"] for r in ela_results]
        ensemble_probs = [(c + l + e) / 3 for c, l, e in zip(cnn_fake_probs, lstm_fake_probs, ela_fake_probs)]
    else:
        ensemble_probs = [(c + l) / 2 for c, l in zip(cnn_fake_probs, lstm_fake_probs)]

    # Overall verdict
    avg_fake = np.mean(ensemble_probs)
    fake_frame_count = sum(1 for p in ensemble_probs if p > threshold)
    fake_percentage = fake_frame_count / n_frames * 100

    verdict = "FAKE" if avg_fake > threshold else "REAL"
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
    ax.axhline(y=threshold, color="gray", linestyle="--", alpha=0.5, label=f"Threshold ({threshold:.2f})")
    ax.fill_between(timestamps, ensemble_probs, threshold,
                    where=[p > threshold for p in ensemble_probs],
                    color="#e63946", alpha=0.15, label="Fake region")
    ax.fill_between(timestamps, ensemble_probs, threshold,
                    where=[p <= threshold for p in ensemble_probs],
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

        # Show Grad-CAM for most suspicious frame (combine CNN + LSTM)
        lstm_map = lstm_results[most_fake_idx]["attention_map"]
        cnn_map = cnn_results[most_fake_idx].get("gradcam")
        combined = (lstm_map + cnn_map) / 2 if cnn_map is not None else lstm_map
        fig3, ax3 = plt.subplots(figsize=(3, 3))
        img_resized = frames[most_fake_idx].resize((224, 224))
        ax3.imshow(img_resized)
        attn_up = np.array(Image.fromarray(
            (combined * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)) / 255.0
        ax3.imshow(attn_up, cmap="jet", alpha=0.45)
        ax3.set_title(f"Grad-CAM: {lstm_results[most_fake_idx]['suspicious_region']}")
        ax3.axis("off")
        st.pyplot(fig3)
        plt.close(fig3)

    with col2:
        st.markdown(f"**Most authentic frame** (#{most_real_idx + 1}, "
                    f"fake prob: {ensemble_probs[most_real_idx] * 100:.1f}%)")
        st.image(frames[most_real_idx], width=300)

        lstm_map2 = lstm_results[most_real_idx]["attention_map"]
        cnn_map2 = cnn_results[most_real_idx].get("gradcam")
        combined2 = (lstm_map2 + cnn_map2) / 2 if cnn_map2 is not None else lstm_map2
        fig4, ax4 = plt.subplots(figsize=(3, 3))
        img_resized2 = frames[most_real_idx].resize((224, 224))
        ax4.imshow(img_resized2)
        attn_up2 = np.array(Image.fromarray(
            (combined2 * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)) / 255.0
        ax4.imshow(attn_up2, cmap="jet", alpha=0.45)
        ax4.set_title("Grad-CAM - authentic frame")
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
        lstm_m = lstm_results[frame_idx]["attention_map"]
        cnn_m = cnn_results[frame_idx].get("gradcam")
        show_map = (lstm_m + cnn_m) / 2 if cnn_m is not None else lstm_m
        fig5, ax5 = plt.subplots(figsize=(3, 3))
        frame_resized = frames[frame_idx].resize((224, 224))
        ax5.imshow(frame_resized)
        map_up = np.array(Image.fromarray(
            (show_map * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)) / 255.0
        ax5.imshow(map_up, cmap="jet", alpha=0.45)
        ax5.set_title("Grad-CAM")
        ax5.axis("off")
        st.pyplot(fig5)
        plt.close(fig5)
    with bcol3:
        fr_verdict = "Fake" if ensemble_probs[frame_idx] > threshold else "Real"
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
    face_detector = load_face_detector()

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("Settings")

    input_type = st.sidebar.radio("Input type", ["Image", "Video"])

    threshold = st.sidebar.slider(
        "Detection threshold",
        min_value=0.30, max_value=0.70, value=0.45, step=0.05,
        help="Lower = more sensitive to fakes. Default 0.45 catches subtle manipulations."
    )

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
            with st.spinner("Detecting face and running analysis..."):
                cnn_result, lstm_result, ela_result, face_crop = analyse_single_image(
                    cnn_model, lstm_model, image, face_detector
                )

            st.session_state["cnn_result"] = cnn_result
            st.session_state["lstm_result"] = lstm_result
            st.session_state["ela_result"] = ela_result
            st.session_state["face_crop"] = face_crop
            st.session_state["mode"] = "image"
            assistant.set_detection_context(cnn_result, lstm_result)
            st.session_state["assistant"] = assistant
            st.session_state["chat_history"] = []

        if st.session_state.get("mode") == "image" and "cnn_result" in st.session_state:
            # Show the detected face crop
            if "face_crop" in st.session_state:
                with st.expander("Detected face (input to model)"):
                    fc1, fc2 = st.columns(2)
                    fc1.image(image, caption="Original", width=250)
                    fc2.image(st.session_state["face_crop"], caption="Cropped face (model input)", width=250)
            show_image_results(
                st.session_state["cnn_result"],
                st.session_state["lstm_result"],
                st.session_state.get("face_crop", image),
                threshold=threshold,
                ela_result=st.session_state.get("ela_result"),
            )

    else:  # Video
        if frames is None:
            st.info("Please upload a video to proceed.")
            return

        if st.button("Analyse Video", type="primary"):
            progress = st.progress(0)
            st.write(f"Analysing {len(frames)} frames...")

            cnn_results, lstm_results, ela_results, face_crops = analyse_video_frames(
                cnn_model, lstm_model, frames, face_detector, progress_bar=progress
            )

            st.session_state["video_cnn_results"] = cnn_results
            st.session_state["video_lstm_results"] = lstm_results
            st.session_state["video_ela_results"] = ela_results
            st.session_state["video_frames"] = face_crops  # use cropped faces for display
            st.session_state["video_fps"] = fps
            st.session_state["mode"] = "video"

            # Set context for LLM using aggregated results
            avg_cnn_fake = np.mean([r["prob_fake"] for r in cnn_results])
            avg_lstm_fake = np.mean([r["prob_fake"] for r in lstm_results])
            avg_ela_fake = np.mean([r["ela_fake_prob"] for r in ela_results])
            fake_count = sum(1 for c, l, e in zip(cnn_results, lstm_results, ela_results)
                           if (c["prob_fake"] + l["prob_fake"] + e["ela_fake_prob"]) / 3 > threshold)

            summary_cnn = {
                "prediction": "Fake" if avg_cnn_fake > threshold else "Real",
                "confidence": max(avg_cnn_fake, 1 - avg_cnn_fake),
                "prob_real": 1 - avg_cnn_fake,
                "prob_fake": avg_cnn_fake,
            }
            summary_lstm = {
                "prediction": "Fake" if avg_lstm_fake > threshold else "Real",
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
                threshold=threshold,
                ela_results=st.session_state.get("video_ela_results"),
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
