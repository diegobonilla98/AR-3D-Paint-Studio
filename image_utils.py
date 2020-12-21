import cv2
import numpy as np
import matplotlib.pyplot as plt


model_img = cv2.imread('model.png', 0)

model_img = cv2.resize(model_img, (580, 580), cv2.INTER_LANCZOS4)

akaze = cv2.AKAZE_create()
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
kp_model, des_model = akaze.detectAndCompute(model_img, None)


def get_board_pos_mat(images, threshold=130, show=False):
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = akaze.detectAndCompute(gray, None)
    matches = matcher.knnMatch(des_model, des_frame, 2)
    good = []
    nn_match_ratio = 0.9
    for m, n in matches:
        if m.distance < nn_match_ratio * n.distance:
            good.append(m)

    if len(good) > threshold:
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if np.around(np.linalg.det(M)) != 1.0:
            return False, None, None, None
        h, w = model_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        if show:
            canvas = images.copy()
            cv2.polylines(canvas, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
            for i, c in enumerate(dst):
                x, y = c[0]
                cv2.circle(canvas, (int(x), int(y)), 10, (0, 255, 0), -1)
                cv2.putText(canvas, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1., (255, 0, 0), 1)
            plt.imshow(canvas[:, :, ::-1]), plt.show()

        return True, pts, dst, M
    return False, None, None, None


lower_ycrcb = np.array([0, 133, 77], dtype="uint8")
upper_ycrcb = np.array([255, 173, 127], dtype="uint8")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def get_hand_mask(coords, org):
    mask = np.zeros_like(org)
    cv2.fillPoly(mask, [np.int32(coords)], (255, 255, 255))

    ycrcb_image = cv2.cvtColor(org.copy(), cv2.COLOR_BGR2YCR_CB)
    skin_region = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)
    skin_region = cv2.morphologyEx(skin_region, cv2.MORPH_OPEN, kernel, iterations=1)
    skin_region = cv2.morphologyEx(skin_region, cv2.MORPH_DILATE, kernel, iterations=1)
    skin_region = cv2.cvtColor(skin_region, cv2.COLOR_GRAY2BGR)

    return cv2.bitwise_and(mask, skin_region)
