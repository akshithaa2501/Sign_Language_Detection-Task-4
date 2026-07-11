import streamlit as st
from PIL import Image
import cv2
import datetime
import tempfile
from predict import predict_image, predict_frame

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 Sign Language Detection System")

st.markdown("""
This application detects sign language using a trained **YOLOv8** model.

### Supported Signs

- Hello
- Yes
- No
- Thank You
- I Love You
""")

# =====================================================
# TIME RESTRICTION
# =====================================================

current_time = datetime.datetime.now().time()

start_time = datetime.time(18, 0)
end_time = datetime.time(22, 0)

if not (start_time <= current_time <= end_time):
    st.error("❌ This application works only between 6:00 PM and 10:00 PM.")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Sign Language Detection")

mode = st.sidebar.radio(
    "Select Mode",
    (
        "Upload Images",
        "Webcam"
    )
)

st.sidebar.markdown("---")
st.sidebar.success("YOLOv8 Model Loaded Successfully")

st.markdown("---")
# =====================================================
# MULTIPLE IMAGE UPLOAD
# =====================================================

if mode == "Upload Images":

    uploaded_files = st.file_uploader(
        "Upload one or more sign language images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(f"{len(uploaded_files)} image(s) selected.")

        for i, uploaded_file in enumerate(uploaded_files):

            st.markdown("---")

            st.subheader(f"Image {i+1}")

            image = Image.open(uploaded_file)

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

            if st.button(
                f"Detect Sign {i+1}",
                key=f"detect_{i}"
            ):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as temp:

                    image.save(temp.name)

                    sign, confidence, result_image = predict_image(temp.name)

                if result_image is not None:

                    st.image(
                        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                        caption="Detection Result",
                        use_container_width=True
                    )

                if sign == "No Sign Detected":

                    st.error("❌ No Sign Detected")

                else:

                    col1, col2 = st.columns(2)

                    with col1:
                        st.success("Detected Sign")
                        st.markdown(f"## {sign}")

                    with col2:
                        st.success("Confidence")
                        st.markdown(f"## {confidence*100:.2f}%")
                        # =====================================================
# WEBCAM DETECTION
# =====================================================

elif mode == "Webcam":

    st.subheader("📷 Real-Time Webcam Detection")

    start = st.button("▶ Start Webcam", key="start_webcam")
    stop = st.button("⏹ Stop Webcam", key="stop_webcam")

    FRAME_WINDOW = st.empty()
    STATUS = st.empty()

    if start:

        camera = cv2.VideoCapture(0)

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)

        while True:

            ret, frame = camera.read()

            if not ret:
                STATUS.error("Unable to access webcam.")
                break

            sign, confidence, result_frame = predict_frame(frame)

            FRAME_WINDOW.image(
                cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

            if sign == "No Sign Detected":

                STATUS.warning("⚠ No Sign Detected")

            else:

                STATUS.success(
                    f"""
### Detected Sign

**{sign}**

Confidence : **{confidence*100:.2f}%**
"""
                )

            if stop:
                break

        camera.release()
        cv2.destroyAllWindows()
        # =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
"""
## About

This Sign Language Detection System is developed using **YOLOv8** and **Streamlit**.

### Features

✅ Multiple Image Upload

✅ Real-Time Webcam Detection

✅ Supported Signs

- Hello
- Yes
- No
- Thank You
- I Love You

✅ Confidence Score

✅ Works only between **6 PM and 10 PM**
"""
)

st.markdown("---")

st.success("✅ Application Ready")

st.sidebar.markdown("---")
st.sidebar.info(
"""
Developed using:

• YOLOv8

• Streamlit

• OpenCV

• Python
"""
)