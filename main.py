from aim_assistant import AimAssistant
from drawer import Drawer
import cv2
import os


N = 5  # run nn every N frames


def process_file(path_to_file: str, save_path: str, assistant: AimAssistant, file_type: str) -> None:
    try:
        cap = cv2.VideoCapture(path_to_file)
    except Exception as e:
        print("Failed while creating a cap. Error: {e}")
        raise e
    if not cap.isOpened():
        raise Exception("Cap's not opened")

    cv2.namedWindow("", cv2.WINDOW_NORMAL)

    if file_type == "video":
        output_name = os.path.join(save_path, os.path.splitext(os.path.basename(path_to_file))[0] + '.avi')
        try:
            video_writter = cv2.VideoWriter(
                output_name, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10,
                (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )
        except Exception as e:
            print("Failed while initializing the video writer. Error: {e}")
            raise e
    else:
        output_name = os.path.join(save_path, os.path.splitext(os.path.basename(path_to_file))[0] + '.jpg')

    # Process file
    frame_counter = 0
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        # Run neural net once in N frames
        if frame_counter % N == 0:
            output = assistant.aim(frame)
            Drawer.visualise_results(output, frame)

        # Save results
        if file_type == "video":
            video_writter.write(frame)
            frame_counter += 1
        else:
            try:
                cv2.imwrite(output_name, frame)
            except Exception as e:
                print(f"Failed while saving a processed photo. Error: {e}")
                raise e

        cv2.imshow("", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    return


def main():
    aim_assistant = AimAssistant(crop=True)
    file_to_process = r"D:\Desktop\Reserve_NNs\Datasets\raw_data\videos_Lemekh\test_trims\100-101_test_1.mp4"
    #file_to_process = r"D:\Desktop\system_output\aim_assistance\test4.JPG"
    save_path = r"D:\Desktop\system_output\aim_assistance\results"

    if any(file_to_process.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]):
        file_type = "photo"
    elif file_to_process.endswith(".mp4"):
        file_type = "video"
    else:
        raise Exception(f'Cannot process the file. Extension: {os.path.splitext(file_to_process)[-1]} is not supported')

    process_file(file_to_process, save_path, aim_assistant, file_type=file_type)


if __name__ == "__main__":
    main()
