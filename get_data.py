from image_utils import *
import uuid

cam = cv2.VideoCapture('point.mp4')
while True:
    _, frame = cam.read()

    frame = cv2.resize(frame, (int(960 * 1.5), int(411 * 1.5)), cv2.INTER_LANCZOS4)
    frame = cv2.flip(frame, -1)

    canvas = frame.copy()
    try:
        status, org, coords, M = get_board_pos_mat(frame)
        if status:
            mask = get_hand(coords, frame)
            if mask is None:
                continue
            cv2.imwrite(f'./train/point/{str(uuid.uuid4())}.jpg', mask)
            # cv2.imshow('Mask', mask)
    except:
        continue

    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     break

cv2.destroyAllWindows()
cam.release()
