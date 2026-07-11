from ultralytics import YOLO
import cv2
import os

# =====================================================
# LOAD MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

model = YOLO(MODEL_PATH)

# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = {
    "hello": "Hello",
    "yes": "Yes",
    "no": "No",
    "thankyou": "Thank You",
    "iloveyou": "I Love You"
}

# =====================================================
# IMAGE PREDICTION
# =====================================================

def predict_image(image_path):

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.01,
        iou=0.45,
        augment=True,
        save=False,
        verbose=False
    )

    best_sign = "No Sign Detected"
    best_conf = 0.0
    annotated_image = None

    for result in results:

        annotated_image = result.plot()

        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf > best_conf:

                best_conf = conf

                class_name = model.names[cls]

                best_sign = CLASS_NAMES.get(
                    class_name,
                    class_name
                )

    return best_sign, best_conf, annotated_image


# =====================================================
# WEBCAM PREDICTION
# =====================================================

def predict_frame(frame):

    # Flip webcam for natural view
    frame = cv2.flip(frame, 1)

    # Resize frame
    frame = cv2.resize(frame, (640, 640))

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.01,
        iou=0.45,
        augment=True,
        save=False,
        verbose=False
    )

    best_sign = "No Sign Detected"
    best_conf = 0.0
    annotated_frame = frame.copy()

    for result in results:

        annotated_frame = result.plot()

        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf > best_conf:

                best_conf = conf

                class_name = model.names[cls]

                best_sign = CLASS_NAMES.get(
                    class_name,
                    class_name
                )

    return best_sign, best_conf, annotated_frame