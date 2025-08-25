import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dental Detection App", page_icon="🦷", layout="wide")

st.title("🦷 Dental Detection App")

with st.container(border=True):
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Cache models
@st.cache_resource
def load_model(path):
    return YOLO(path)

# Paths
detect_model_path = "runs/detect/train/weights/best.pt"
classify_model1_path = "classify/classify.pt"
classify_model2_path = "classify/leftRightFront.pt"

# Load models
detect_model = load_model(detect_model_path)
classify_model1 = load_model(classify_model1_path)
classify_model2 = load_model(classify_model2_path)

# Detection class names
class_names = [
    "Cavity", "Missing", "Permanent teeth",
    "Restoration", "Root stump", "Stainless steel crown",
    "Supernumerary teeth", "Teeths",
    "edentulous space", "void"
]

# Colors
colors = {
    "Cavity": (255, 0, 0),
    "Missing": (0, 255, 0),
    "Permanent teeth": (0, 0, 255),
    "Restoration": (255, 165, 0),
    "Root stump": (128, 0, 128),
    "Stainless steel crown": (0, 255, 255),
    "Supernumerary teeth": (255, 20, 147),
    "Teeths": (70, 130, 180),
    "edentulous space": (139, 69, 19),
    "void": (192, 192, 192)
}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    base_image = np.array(image).copy()
    
    st.subheader("🔍 Detection and classification")
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 2, 1])

        # ---------- Column 1: Checklist ----------
        with col1:
            with st.container(border=True, height=516):
                st.write("Select classes to detect:")
                selected_classes = []
                for cname in class_names:
                    if st.checkbox(cname, value=False, key=f"chk_{cname}"):
                        selected_classes.append(cname)

        # ---------- Run detection ----------
        results = detect_model.predict(image)
        all_detection_counts = {}
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = class_names[cls_id]
                all_detection_counts[label] = all_detection_counts.get(label, 0) + 1

        # Prepare display image
        if selected_classes:
            display_img = base_image.copy()
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = class_names[cls_id]
                    if label in selected_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = colors.get(label, (0, 100, 225))

                        # Semi-transparent fill
                        overlay = display_img.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        alpha = 0.2
                        display_img = cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0)

                        # Border
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 3)

                        # ====== Add Label Text (relative to image size) ======
                        text = f"{label}"
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        # Scale font dynamically based on image size
                        img_h, img_w = display_img.shape[:2]
                        ref_w, ref_h = 2344, 1825   # reference resolution you trained on (adjust if needed)
                        ref_font_scale = 0.8
                        scale_factor = ((img_w * img_h) / (ref_w * ref_h)) ** 0.5
                        font_scale = ref_font_scale * scale_factor
                        font_scale = max(0.5, min(font_scale, 2.0))  # clamp between 0.5 and 2

                        thickness = max(1, int(font_scale * 2))

                        # Text size
                        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                        # Background rectangle
                        cv2.rectangle(display_img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

                        # Put label text
                        cv2.putText(
                            display_img,
                            text,
                            (x1 + 3, y1 - 5),
                            font,
                            font_scale,
                            (255, 255, 255),  # white text
                            thickness
                        )

        else:
            display_img = base_image

        # ---------- Column 2: Always preview ----------
        with col2:
            with st.container(border=True, height=517):
                st.image(display_img, caption="Uploaded Image", use_container_width=False)

        # ---------- Column 3: Classification ----------
        combined_classifications = {}
        with col3:
            with st.container(border=True, height=516):
                st.subheader("🧾 Classifications")

                # Run both models
                for model in [classify_model1, classify_model2]:
                    class_results = model.predict(image)
                    for r in class_results:
                        top_indices = r.probs.top5
                        top_probs = [r.probs.data[i].item() for i in top_indices]
                        top_labels = [r.names[i] for i in top_indices]
                        for lbl, prob in zip(top_labels, top_probs):
                            combined_classifications[lbl] = max(
                                combined_classifications.get(lbl, 0), prob
                            )

                # Display combined results
                for lbl, prob in sorted(combined_classifications.items(), key=lambda x: x[1], reverse=True):
                    st.success(f"**{lbl}** ({prob*100:.2f}%)")

    # ---------- Summary Section ----------
    st.subheader("📊 Summary")
    with st.container(border=True):
        colA, colB = st.columns(2)

        with colA: 
            st.write("### Combined Classification Confidence")
            fig, ax = plt.subplots(figsize=(5, 5))
            labels = list(combined_classifications.keys())
            probs = [p*100 for p in combined_classifications.values()]

            bars = ax.bar(labels, probs, color="skyblue")
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Classification Predictions")
            plt.xticks(rotation=30, ha="right")

            # ✅ Add percentage labels on top of bars
            ax.bar_label(bars, labels=[f"{val:.2f}%" for val in probs], padding=3)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)


        with colB:
            if all_detection_counts:
                st.write("### Detection Counts (All)")
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                values = list(all_detection_counts.values())
                labels = list(all_detection_counts.keys())
                def absolute_count(pct, all_vals):
                    total = sum(all_vals)
                    count = int(round(pct*total/100.0))
                    return f"{count}"
                ax2.pie(
                    values,
                    labels=labels,
                    autopct=lambda pct: absolute_count(pct, values)
                )
                ax2.set_title("Detected Classes Distribution")
                ax2.set_aspect('equal')
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)

else:
    st.info("👆 Upload an image to get started.")
