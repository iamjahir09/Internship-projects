import cv2

from processor import DrowsinessProcessor


class VideoProcessor:
    def __init__(self):
        self.processor = DrowsinessProcessor()

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            raise ValueError(
                f"Could not create output video: {output_path}"
            )

        total_frames = 0
        max_sleeping_count = 0
        last_sleeping_ages = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            result = self.processor.process_image(frame)

            processed_frame = result["image"]

            writer.write(processed_frame)

            current_sleeping = result["sleeping_count"]

            if current_sleeping > max_sleeping_count:
                max_sleeping_count = current_sleeping
                last_sleeping_ages = result["sleeping_ages"]

            total_frames += 1

        cap.release()
        writer.release()

        return {
            "output_path": output_path,
            "total_frames": total_frames,
            "max_sleeping_count": max_sleeping_count,
            "sleeping_ages": last_sleeping_ages
        }