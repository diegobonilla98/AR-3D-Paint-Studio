import cv2
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import numpy as np

from image_utils import get_board_pos_mat, get_hand

PAINT_BUFFER = []
GESTURES = ["fist", "palm", "point"]

BUFFER_SIZE = 5
BUFFER_IDX = 0
BUFFER_TOTAL_X = 0
BUFFER_TOTAL_Y = 0
buffer_x_coords = [0] * BUFFER_SIZE
buffer_y_coords = [0] * BUFFER_SIZE

cam = cv2.VideoCapture('./images/VID_20201221_134527.mp4')
f = 0
while True:
    ret, frame = cam.read()
    f += 1
    if f % 1 != 0:
        continue
    f = 0

    frame = cv2.resize(frame, (int(960 * 1.5), int(411 * 1.5)), cv2.INTER_LANCZOS4)
    frame = cv2.flip(frame, -1)

    canvas = frame.copy()
    try:
        status, org, coords, M = get_board_pos_mat(frame)
        if status:
            mask, what, pts = get_hand(coords, frame)
            cv2.fillPoly(canvas, [np.int32(coords)], (50, 50, 50))
            canvas = np.where(mask == 255, frame, canvas)

            cv2.putText(canvas, GESTURES[what] if what != -1 else "nothing",
                        (20, 40), cv2.FONT_HERSHEY_PLAIN, 3., (255, 0, 0), 3)
            if what == 2:
                p = min(pts, key=lambda x: x[0] + x[1])

                BUFFER_TOTAL_X -= buffer_x_coords[BUFFER_IDX]
                BUFFER_TOTAL_Y -= buffer_y_coords[BUFFER_IDX]

                buffer_x_coords[BUFFER_IDX] = p[0]
                buffer_y_coords[BUFFER_IDX] = p[1]

                BUFFER_TOTAL_X += buffer_x_coords[BUFFER_IDX]
                BUFFER_TOTAL_Y += buffer_y_coords[BUFFER_IDX]

                BUFFER_IDX += 1

                if BUFFER_IDX >= BUFFER_SIZE:
                    BUFFER_IDX = 0

                p_norm = [0, 0]
                p_norm[0] = BUFFER_TOTAL_X // BUFFER_SIZE
                p_norm[1] = BUFFER_TOTAL_Y // BUFFER_SIZE

                path = mpltPath.Path(coords.reshape((4, 2)))
                is_inside = path.contains_points([p_norm])[0]
                if is_inside:
                    PAINT_BUFFER.append(p_norm)
            elif what == 0 or what == 1:
                p = np.mean(np.array(pts), axis=0)
                cv2.circle(canvas, (int(p[0]), int(p[1])), 20, (0, 255, 0), -1)
            for pt in range(len(PAINT_BUFFER)):
                if pt == len(PAINT_BUFFER) - 1:
                    break
                cv2.line(canvas, (PAINT_BUFFER[pt][0], PAINT_BUFFER[pt][1]),
                         (PAINT_BUFFER[pt+1][0], PAINT_BUFFER[pt+1][1]), (170, 170, 170), 2)

    except (cv2.error, ValueError, np.linalg.LinAlgError) as e:
        print(e)
        pass

    cv2.imshow('Result', canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()

