import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)


def distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_hand_gesture(image):
    # error -> -1, None, None
    # pointing -> 0, index coords, None
    # fist -> 1, None, None
    # open hand -> 2, median coords, (success, rotation, translation)

    image_rows, image_cols = image.shape[:2]

    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return -1, None, None
    hand_landmarks = results.multi_hand_landmarks[0]

    landmarks = [mp_drawing._normalized_to_pixel_coordinates(l.x, l.y, image_cols, image_rows) for l in
                 hand_landmarks.landmark]
    if not all(landmarks):
        return -1, None, None
    landmarks = np.array(landmarks)
    hand_w = max(landmarks, key=lambda x: x[0])[0] - min(landmarks, key=lambda x: x[0])[0]
    hand_h = max(landmarks, key=lambda x: x[1])[1] - min(landmarks, key=lambda x: x[1])[1]
    median_center = ((landmarks[0][0] + landmarks[13][0]) / 2, (landmarks[0][1] + landmarks[13][1]) / 2)

    d = distance((landmarks[8][0] / hand_w, landmarks[8][1] / hand_h),
                 (median_center[0] / hand_w, median_center[1] / hand_h)) > 0.85
    if d:
        d = distance((landmarks[12][0] / hand_w, landmarks[12][1] / hand_h),
                     (median_center[0] / hand_w, median_center[1] / hand_h)) > 0.7
        if d:
            return 2, median_center, 'blablablalaskdlaksd'
        else:
            return 0, landmarks[8], None
    else:
        return 1, None, None

    # cv2.circle(image, (int(median_center[0]), int(median_center[1])), 10, (0, 255, 0), -1)
    # for idx, landmark in enumerate(landmarks):
    #     if landmark is None:
    #         continue
    #     cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0), -1)
    #     cv2.putText(image, str(idx), (landmark[0] + 20, landmark[1] + 20), cv2.FONT_HERSHEY_COMPLEX,
    #                 0.5, (255, 0, 0), 1)
    # cv2.imshow('Result', image)


# while True:
#     _, frame = cam.read()
#
#     get_hand_gesture(frame)
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#
# cv2.destroyAllWindows()
# cam.release()

