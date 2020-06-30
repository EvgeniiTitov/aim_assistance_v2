import cv2
import numpy as np


class Drawer:

    @staticmethod
    def visualise_results(predictions: dict, frame: np.ndarray) -> None:
        frame_centre = frame.shape[1] // 2, frame.shape[0] // 2
        for index, prediction in predictions.items():
            obj_class = prediction["obj_class"]
            bb_centre = prediction["bb_centre"]
            bb_coord = prediction["bb_coord"]
            angles = prediction["aim_angle"]
            obj_size = prediction["bb_size"]

            cv2.rectangle(
                frame, (bb_coord[0], bb_coord[1]), (bb_coord[2], bb_coord[3]), (0, 255, 0), thickness=4
            )
            cv2.line(frame, frame_centre, bb_centre, (255, 0, 0), thickness=3)

            name = "insulator" if obj_class == 0 else "dumper"
            metadata = f"name:{name}, angle:{angles}, diagonal:{obj_size}"
            cv2.putText(
                frame, metadata, (bb_coord[0], bb_coord[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )

        return
