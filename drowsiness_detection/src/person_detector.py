import os

from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_path=None):

        src_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        project_dir = os.path.dirname(src_dir)

        if model_path is None:

            model_path = os.path.join(
                project_dir,
                "yolo11n.pt"
            )

        model_path = os.path.abspath(
            model_path
        )

        self.model = YOLO(model_path)

    def detect(self, image):

        results = self.model(
            image,
            verbose=False
        )

        people = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                if class_id != 0:
                    continue

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                people.append({
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    "confidence": confidence
                })

        return people