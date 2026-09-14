import os
import urllib.request

import cv2


class FaceDetector:

    def __init__(self):

        src_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        project_dir = os.path.dirname(src_dir)

        model_dir = os.path.join(
            project_dir,
            "model"
        )

        os.makedirs(
            model_dir,
            exist_ok=True
        )

        self.model_path = os.path.join(
            model_dir,
            "face_detection_yunet_2023mar.onnx"
        )

        if not os.path.exists(self.model_path):

            url = (
                "https://github.com/opencv/opencv_zoo/"
                "raw/main/models/face_detection_yunet/"
                "face_detection_yunet_2023mar.onnx"
            )

            try:

                urllib.request.urlretrieve(
                    url,
                    self.model_path
                )

            except Exception as e:

                raise RuntimeError(
                    f"Could not download face detector: {e}"
                )

        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )

    def detect(self, image):

        if image is None:
            return []

        if image.size == 0:
            return []

        h, w = image.shape[:2]

        self.detector.setInputSize(
            (w, h)
        )

        _, detections = self.detector.detect(
            image
        )

        if detections is None:
            return []

        faces = []

        for detection in detections:

            x = int(detection[0])
            y = int(detection[1])
            fw = int(detection[2])
            fh = int(detection[3])

            confidence = float(
                detection[14]
            )

            x1 = max(
                0,
                x
            )

            y1 = max(
                0,
                y
            )

            x2 = min(
                w,
                x + fw
            )

            y2 = min(
                h,
                y + fh
            )

            if x2 <= x1 or y2 <= y1:
                continue

            faces.append({
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "confidence": confidence
            })

        faces.sort(
            key=lambda face: face["confidence"],
            reverse=True
        )

        return faces

    def detect_eyes(self, face_image):

        if face_image is None:
            return []

        if face_image.size == 0:
            return []

        gray = cv2.cvtColor(
            face_image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(
            gray
        )

        cascade_path = os.path.join(
            os.path.dirname(cv2.__file__),
            "data",
            "haarcascade_eye.xml"
        )

        if not os.path.exists(cascade_path):
            return []

        eye_cascade = cv2.CascadeClassifier(
            cascade_path
        )

        if eye_cascade.empty():
            return []

        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(15, 15)
        )

        result = []

        for x, y, w, h in eyes:

            result.append({
                "bbox": (
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h)
                ),
                "area": int(w * h)
            })

        result.sort(
            key=lambda eye: eye["area"],
            reverse=True
        )

        return result