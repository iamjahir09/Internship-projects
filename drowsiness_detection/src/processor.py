import cv2

from person_detector import PersonDetector
from face_detector import FaceDetector
from drowsiness import DrowsinessDetector
from age_detector import AgeDetector


class DrowsinessProcessor:

    def __init__(self):
        self.person_detector = PersonDetector()
        self.face_detector = FaceDetector()
        self.drowsiness_detector = DrowsinessDetector()
        self.age_detector = AgeDetector()

    def draw_label(self, image, text, x, y, color):

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        h, w = image.shape[:2]

        x = max(
            5,
            min(x, w - text_width - 15)
        )

        y = max(
            text_height + 15,
            min(y, h - 5)
        )

        cv2.rectangle(
            image,
            (
                x,
                y - text_height - 12
            ),
            (
                x + text_width + 10,
                y + baseline
            ),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            image,
            text,
            (
                x + 5,
                y - 5
            ),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    def process_image(self, image):

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        output = image.copy()

        h, w = image.shape[:2]

        people = self.person_detector.detect(
            image
        )

        total_people = len(people)

        sleeping_count = 0
        awake_count = 0
        unknown_count = 0

        sleeping_ages = []

        person_results = []

        for person in people:

            x1, y1, x2, y2 = person["bbox"]

            person_confidence = (
                person["confidence"] * 100
            )

            x1 = max(
                0,
                min(x1, w - 1)
            )

            y1 = max(
                0,
                min(y1, h - 1)
            )

            x2 = max(
                0,
                min(x2, w)
            )

            y2 = max(
                0,
                min(y2, h)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            box_width = x2 - x1
            box_height = y2 - y1

            pad_x = int(
                box_width * 0.08
            )

            pad_y = int(
                box_height * 0.08
            )

            px1 = max(
                0,
                x1 - pad_x
            )

            py1 = max(
                0,
                y1 - pad_y
            )

            px2 = min(
                w,
                x2 + pad_x
            )

            py2 = min(
                h,
                y2 + pad_y
            )

            person_crop = image[
                py1:py2,
                px1:px2
            ]

            faces = self.face_detector.detect(
                person_crop
            )

            if len(faces) == 0:

                unknown_count += 1

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    3
                )

                self.draw_label(
                    output,
                    f"Unknown | {person_confidence:.1f}%",
                    x1,
                    y1 + 30,
                    (255, 255, 0)
                )

                person_results.append({
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "state": "Unknown",
                    "age": None,
                    "confidence": person_confidence
                })

                continue

            best_face = max(
                faces,
                key=lambda face:
                face.get(
                    "confidence",
                    0
                )
            )

            fx1, fy1, fx2, fy2 = (
                best_face["bbox"]
            )

            face_x1 = px1 + fx1
            face_y1 = py1 + fy1
            face_x2 = px1 + fx2
            face_y2 = py1 + fy2

            face_x1 = max(
                0,
                min(face_x1, w - 1)
            )

            face_y1 = max(
                0,
                min(face_y1, h - 1)
            )

            face_x2 = max(
                0,
                min(face_x2, w)
            )

            face_y2 = max(
                0,
                min(face_y2, h)
            )

            if (
                face_x2 <= face_x1
                or
                face_y2 <= face_y1
            ):

                unknown_count += 1

                person_results.append({
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "state": "Unknown",
                    "age": None,
                    "confidence": 0
                })

                continue

            face = image[
                face_y1:face_y2,
                face_x1:face_x2
            ]

            if face is None or face.size == 0:

                unknown_count += 1

                person_results.append({
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "state": "Unknown",
                    "age": None,
                    "confidence": 0
                })

                continue

            drowsiness = (
                self.drowsiness_detector.predict(
                    face
                )
            )

            age = self.age_detector.predict(
                face
            )

            if drowsiness is None:

                unknown_count += 1

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    3
                )

                self.draw_label(
                    output,
                    "Unknown",
                    x1,
                    y1 + 30,
                    (255, 255, 0)
                )

                person_results.append({
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "face_bbox": (
                        face_x1,
                        face_y1,
                        face_x2,
                        face_y2
                    ),
                    "state": "Unknown",
                    "age": age,
                    "confidence": 0
                })

                continue

            state = drowsiness["state"]

            confidence = (
                drowsiness["confidence"]
            )

            if state == "Sleeping":

                sleeping_count += 1

                if age is not None:
                    sleeping_ages.append(
                        age
                    )

                box_color = (
                    0,
                    0,
                    255
                )

            elif state == "Awake":

                awake_count += 1

                box_color = (
                    0,
                    255,
                    0
                )

            else:

                unknown_count += 1

                box_color = (
                    255,
                    255,
                    0
                )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )

            cv2.rectangle(
                output,
                (face_x1, face_y1),
                (face_x2, face_y2),
                box_color,
                2
            )

            age_text = (
                str(age)
                if age is not None
                else "Unknown"
            )

            label = (
                f"{state} | Age: {age_text} | "
                f"{confidence:.1f}%"
            )

            label_y = y1 + 30

            self.draw_label(
                output,
                label,
                x1,
                label_y,
                box_color
            )

            person_results.append({
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "face_bbox": (
                    face_x1,
                    face_y1,
                    face_x2,
                    face_y2
                ),
                "state": state,
                "age": age,
                "confidence": confidence
            })

        summary_height = 130

        summary_top = max(
            10,
            h - summary_height
        )

        summary_width = 340

        cv2.rectangle(
            output,
            (
                10,
                summary_top
            ),
            (
                summary_width,
                h - 10
            ),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            output,
            f"Total People: {total_people}",
            (
                25,
                summary_top + 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"Sleeping: {sleeping_count}",
            (
                25,
                summary_top + 60
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"Awake: {awake_count}",
            (
                25,
                summary_top + 90
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            f"Unknown: {unknown_count}",
            (
                25,
                summary_top + 120
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        return {
            "image": output,
            "total_people": total_people,
            "sleeping_count": sleeping_count,
            "awake_count": awake_count,
            "unknown_count": unknown_count,
            "sleeping_ages": sleeping_ages,
            "people": person_results
        }