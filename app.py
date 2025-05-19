import streamlit as st
from PIL import Image
from model import EnsembleDetector
import uuid
import os

# Initialize model once
model = EnsembleDetector()

# App title
st.title("🔍 Object Detection Web App")

# Upload UI
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display input image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # Run model
    with st.spinner("Detecting objects..."):
        result_image = model.predict(image.copy(), return_image=True)
        boxes, scores, labels = model.predict(image.copy(), return_image=False)

        # Save output image
        os.makedirs("static/uploads", exist_ok=True)
        filename = f"static/uploads/result_{uuid.uuid4().hex}.jpg"
        result_image.save(filename)

        # Show output
        st.image(result_image, caption="Detection Result", use_column_width=True)
        st.markdown("### 🎯 Detected Objects")
        for label, score in zip(labels, scores):
            label_name = model.names[int(label)]
            st.markdown(f"- **{label_name}**: `{score * 100:.2f}%`")
