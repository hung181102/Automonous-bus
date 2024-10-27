import serial
import argparse
import sys
import time
import numpy as np
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
# import utils
_MARGIN = -200  # pixels
_ROW_SIZE = -10  # pixels
_FONT_SIZE = 2
_FONT_THICKNESS = 3
_TEXT_COLOR = (0, 255, 255)  # red
def visualize(image: np.ndarray, detection_result: processor.DetectionResult) -> np.ndarray:
    category_name = ""
    bbox_height = 0
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + bbox.origin_x, _MARGIN + _ROW_SIZE + bbox.origin_y)
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)
        cv2.putText(image, result_text, (20, 90), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200), _FONT_THICKNESS)
        # cv2.putText(image, category_name, text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200), _FONT_THICKNESS)
        bbox_height = bbox.height
    return image, category_name, bbox_height
ser = serial.Serial('/dev/ttyACM0', 9600)
def nothing(x):
    pass
_MARGIN = -20  # pixels
_ROW_SIZE = -10  # pixels
_FONT_SIZE = 2
_FONT_THICKNESS = 3
_TEXT_COLOR = (0, 255, 255)  # red
def run(model: str, camera_id: int, width: int, height: int, num_threads: int, enable_edgetpu: bool) -> None:
    cv2.namedWindow("Trackbars")
    cv2.namedWindow("Phase")
    cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 101, 255, nothing)
    cv2.createTrackbar("x1", "Phase", 108, 640, nothing)
    cv2.createTrackbar("y1", "Phase", 359, 480, nothing)
    cv2.createTrackbar("x2", "Phase", 65, 640, nothing)
    cv2.createTrackbar("y2", "Phase", 472, 480, nothing)
    cv2.createTrackbar("x3", "Phase", 425, 640, nothing)
    cv2.createTrackbar("y3", "Phase", 359, 480, nothing)
    cv2.createTrackbar("x4", "Phase", 441, 640, nothing)
    cv2.createTrackbar("y4", "Phase", 472, 480, nothing)
    counter, fps = 0, 0
    start_time = time.time()
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    row_size = 20
    left_margin = 24
    text_color = (0, 0, 255)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    base_options = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.9)
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)
    prev_direction = None
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')
        counter += 1
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        detection_result = detector.detect(input_tensor)
        image, category_name, bbox_height = visualize(image, detection_result)
        image = cv2.resize(image, (640, 480))
        #  print(category_name)
        #print(bbox_height)
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        x1 = cv2.getTrackbarPos("x1", "Phase")
        x2 = cv2.getTrackbarPos("x2", "Phase")
        x3 = cv2.getTrackbarPos("x3", "Phase")
        x4 = cv2.getTrackbarPos("x4", "Phase")
        y1 = cv2.getTrackbarPos("y1", "Phase")
        y2 = cv2.getTrackbarPos("y2", "Phase")
        y3 = cv2.getTrackbarPos("y3", "Phase")
        y4 = cv2.getTrackbarPos("y4", "Phase")
        tl, bl, tr, br = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
        cv2.circle(image, tl, 5, (0, 0, 255), -1)
        cv2.circle(image, bl, 5, (0, 0, 255), -1)
        cv2.circle(image, tr, 5, (0, 0, 255), -1)
        cv2.circle(image, br, 5, (0, 0, 255), -1)
        tl, bl, tr, br = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(image, matrix, (640, 480))
        hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv_transformed_frame, lower, upper)
        histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint
        #if left_base > 280 or right_base < 380:
            #mid_base = (left_base + 640) // 2
            # elif left_base == 0:
            # mid_base = 476
        if right_base == 320:
            mid_base = (left_base + 640) // 2
        else:
            mid_base = (left_base + right_base) // 2

        angle_difference = mid_base - 320
        angle_radians = np.arctan2(angle_difference, 480)
        angle_degrees = np.degrees(angle_radians)
        rounded_angle = round(angle_degrees, 2)
        print(left_base)
        print(right_base)
        cv2.putText(image, "curve: {:.2f}".format(rounded_angle), (20, 50), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE,
                    (200, 0, 200), _FONT_THICKNESS)
        y = 472
        lx = []
        rx = []
        msk = mask.copy()
        while y > 0:
            img_left = mask[y - 40:y, left_base - 50:left_base + 50]
            contours, _ = cv2.findContours(img_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    lx.append(left_base - 50 + cx)
                    left_base = left_base - 50 + cx

            img_right = mask[y - 40:y, right_base - 50:right_base + 50]
            contours, _ = cv2.findContours(img_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    rx.append(right_base - 50 + cx)
                    right_base = right_base - 50 + cx

            cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 2)
            cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
            # cv2.line(image, (304, 472), (mid_base, 380), (0, 0, 255), 2)
            y -= 40
            direction = None
            if category_name == "bus stop"  and  96 < bbox_height < 99:
                direction = 5
                cv2.putText(image, "Stop", (240, 460), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200),
                            _FONT_THICKNESS)
            elif category_name == "bus"  and 160 < bbox_height < 165:
                direction = 7
                cv2.putText(image, "Go", (240, 460), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200),
                            _FONT_THICKNESS)
            elif category_name == "red light" and bbox_height > 73:
                direction = 9
                cv2.putText(image, "Stop", (240, 460), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200),
                            _FONT_THICKNESS)
            elif category_name == "stop" and bbox_height > 110:
                direction = 9
                cv2.putText(image, "Stop", (240, 460), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200),
                            _FONT_THICKNESS)
            elif category_name == "person" and bbox_height > 300:
                cv2.putText(image, "Stop", (240, 460), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200),
                            _FONT_THICKNESS)
            elif category_name in ["turn right"] and 70 < bbox_height < 72:
                direction = 8
                cv2.putText(image, "Turn Right", (240, 460), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200),
                            _FONT_THICKNESS)
            else:
                if rounded_angle > 5:
                    direction = 4
                    cv2.putText(image, "Turn right", (240, 460), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200),
                                _FONT_THICKNESS)
                elif rounded_angle < -5:
                    direction = 3
                    cv2.putText(image, "Turn left", (240, 460), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200),
                                _FONT_THICKNESS)
                else:
                    direction = 1
                    cv2.putText(image, "Go straight", (240, 460), cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (0, 0, 200),
                                _FONT_THICKNESS)
            if direction != prev_direction:
                prev_direction = direction
                ser.write(str(direction).encode())
                # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow("Original", image)
        #cv2.imshow("Bird's Eye View", transformed_frame)
        #cv2.imshow("Lane Detection - Image Thresholding", mask)
        cv2.imshow("Lane Detection - Sliding Windows", msk)
    cap.release()
    cv2.destroyAllWindows()
def main():
    model_path = 'best.tflite'
    camera_id = 0
    frame_width = 640
    frame_height = 480
    num_threads = 4
    enable_edgetpu = False
    run(model_path, camera_id, frame_width, frame_height, num_threads, enable_edgetpu)
if __name__ == '__main__':
    main()
