import cv2
from ultralytics import YOLO
from color_classifier import predict_color

model = YOLO("yolo11s.pt")


def detect(image_path):
    image = cv2.imread(image_path)

    results = model(
        image_path,
        conf=0.15,
        imgsz=1280
    )

    car_count = 0
    person_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            if class_name == "person":
                person_count += 1

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    "Person",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            elif class_name == "car":
                car_count += 1

                car_crop = image[y1:y2, x1:x2]

                if car_crop.size == 0:
                    continue

                car_crop_rgb = cv2.cvtColor(
                    car_crop,
                    cv2.COLOR_BGR2RGB
                )

                predicted_color, confidence = predict_color(
                    car_crop_rgb
                )

                if predicted_color.lower() == "blue":
                    box_color = (0, 0, 255)
                else:
                    box_color = (255, 0, 0)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                label = f"{predicted_color} {confidence:.2f}"

                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

    cv2.imwrite(
        "Project_1/src/traffic_result.jpg",
        image
    )

    return image, car_count, person_count


if __name__ == "__main__":
    result_image, cars, people = detect(
        "Project_1/src/16-traffic-1.webp"
    )

    print("Cars:", cars)
    print("People:", people)