import cv2
import matplotlib.pyplot as plt
import numpy as np

from image_utils import get_board_pos_mat, get_hand_mask


cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()

    canvas = frame.copy()
    try:
        status, org, coords, M = get_board_pos_mat(frame)
        if status:
            mask = get_hand_mask(coords, frame)
            cv2.fillPoly(canvas, [np.int32(coords)], (50, 50, 50))
            canvas = np.where(mask == 255, frame, canvas)


    except (cv2.error, ValueError, np.linalg.LinAlgError) as e:
        print(e)
        pass

    cv2.imshow('Result', canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()

