import numpy as np
import math
from nn import NeuralNet
from typing import List, Dict, Tuple


class AimAssistant:
    """
    Wrapper around the neural net for data preprocessing / postprocessing
    """
    sides_crop = 0.3
    top_crop = 0.35

    def __init__(self, crop: bool = False):
        self.to_crop = crop
        try:
            self.net = NeuralNet()
        except Exception as e:
            print(f"Failed while NN initialization. Error: {e}")
            raise e
        print("Aim assistant successfully initialized")

    def aim(self, frame: np.ndarray) -> dict:
        """

        :param frame:
        :return:
        """
        # Режем кадр, чтобы оставить только центр картинки где располагается опора.
        if self.to_crop:
            relative_coord, frame = AimAssistant.crop_frame(frame, self.sides_crop, self.top_crop)

        # Detect components on the frame
        components = self.net.predict(frame)

        # If components, calculate angles relatively to the frame centre
        output = dict()
        if components:
            # Пересчитаем координаты коробок относительно целого кадра. Сейчас они относительно
            # вырезанного np.ndarray если стоял флаг self.to_crop
            if self.to_crop:
                components = AimAssistant.recalculate_boxes(components, relative_coord)
            output = AimAssistant.calculate_angles(components=components, frame=frame)

        return output

    @staticmethod
    def recalculate_boxes(predictions:List[list], sliced_image_coord: List[int]) -> List[list]:
        """
        Пересчитывает координаты коробки относительно оригинального кадра, а не вырезанной подкартинки
        :param predictions:
        :param sliced_image_coord:
        :return:
        """
        output = list()
        for prediction in predictions:
            class_id = prediction[0]
            conf = prediction[1]
            left = prediction[2] + sliced_image_coord[0]
            top = prediction[3] + sliced_image_coord[1]
            right = prediction[4] + sliced_image_coord[0]
            bot = prediction[5] + sliced_image_coord[1]
            assert all(e > 0 for e in [left, top, right, bot]), "Coordinates recalculated wrong"
            assert all((left < right, top < bot)), "Coordinates recalculated wrong"
            output.append([class_id, conf, left, top, right, bot])

        return output

    @staticmethod
    def crop_frame(frame: np.ndarray, sides: float = 0.3, top: float = 0.35) -> Tuple[list, np.ndarray]:
        """
        Вырезает картинки из кадра где находится опора. Отрезает края и верхушку кадра. Они нам не интеренсты
        если снимаем с курсовой камеры, которая смотрит перед собой.
        :param frame:
        :param sides:
        :param top:
        :return:
        """
        assert 0 < sides < 0.5, "Sides crop needs to be between 0 and 50%"
        assert 0 < top < 0.5, "Top crop needs to be between 0 and 50%"

        frame_height, frame_width = frame.shape[0], frame.shape[1]
        new_left = int(frame_width * sides)
        new_right = int(frame_width - (frame_width * sides))
        new_top = int(frame_height * top)
        new_bot = frame_height
        assert all((new_top < new_bot, new_left < new_right)), "Wrong cropped coordinates"

        #cv2.rectangle(frame, (new_left, new_top), (new_right, new_bot), (0, 255, 255), 2)
        try:
            cropped_image = np.array(frame[new_top:new_bot, new_left:new_right, :])
        except Exception as e:
            print("Failed while cropping the frame. Error: {e}")
            raise e

        return [new_left, new_top, new_right, new_bot], cropped_image

    @staticmethod
    def calculate_angles(components: List[list], frame: np.ndarray) -> Dict[int, dict]:
        """
        :param components: detected components
        :param frame: frame
        :return:
        """
        frame_centre = (frame.shape[1] // 2, frame.shape[0] // 2)  # x,y
        output = dict()
        output_schema = {
            "obj_class": int,
            "bb_centre": tuple,
            "bb_coord": list,
            "aim_angle": tuple,
            "bb_size": float
        }

        for i, component in enumerate(components):
            obj_class, conf = component[0], component[1]
            # Skip concrete pillars, we're interested in only insulators and dumpers
            if obj_class == 2:
                continue
            left, top = component[2], component[3]
            right, bot = component[4], component[5]

            # Calculate element's BB centre relatively to the whole image
            element_x_centre = (left + right) // 2
            element_y_centre = (top + bot) // 2

            # Calculate delta (image centre vs element centre)
            delta_x = abs(frame_centre[0] - element_x_centre)
            delta_y = abs(frame_centre[1] - element_y_centre)

            # Calculate angles
            angle_1 = round(np.rad2deg(np.arctan2(delta_x, delta_y)), 2)
            angle_2 = round(90 - angle_1, 2)
            assert all((0 <= angle_1 <= 90, 0 <= angle_2 <= 90)), "Wrong angles calculated. Expected: [0, 90]"

            # Estimate object's relative diameter
            element_BB_diagonal = math.sqrt((top - bot)**2 +(right - left)**2)
            frame_diagonal = math.sqrt((frame.shape[0])**2 + (frame.shape[1])**2)
            diagonal = round(element_BB_diagonal / frame_diagonal, 3)

            schema_instance = output_schema.copy()
            try:
                schema_instance["obj_class"] = obj_class
                schema_instance["bb_centre"] = (int(element_x_centre), int(element_y_centre))
                schema_instance["bb_coord"] = (left, top, right, bot)
                schema_instance["aim_angle"] = (angle_1, angle_2)
                schema_instance["bb_size"] = diagonal
            except Exception as e:
                print(f"Failed while filling the output schema instance. Error: {e}")
                raise e

            output[i] = schema_instance

        return output
