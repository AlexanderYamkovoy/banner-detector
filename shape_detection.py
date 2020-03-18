import numpy as np
import cv2

four_cc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('old video shape detection3.avi', four_cc, 30, (1280, 720), True)

for i in range(0, 603):
    img = cv2.imread('/Users/oleksandr/Folder/Banner_Detector2.0/old_frames/frame{}.png'.format(i))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('123.png')
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(10000)

    kp1, des1 = sift.detectAndCompute(gray_template, None)
    kp2, des2 = sift.detectAndCompute(gray_img, None)

    index_params = {'algorithm': 1, 'trees': 5}
    search_params = {'checks': 70}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    cr_frame = None
    frame_hsv = None
    min_max = None
    if len(good) >= 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = gray_template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, m)
        x_corner_list = [dst[i][0][0] for i in range(len(dst))]
        y_corner_list = [dst[j][0][1] for j in range(len(dst))]
        x_min, x_max = np.int64(min(x_corner_list)), np.int64(max(x_corner_list))
        y_min, y_max = np.int64(min(y_corner_list)), np.int64(max(y_corner_list))
        min_max = [x_min, x_max, y_min, y_max]
        cr_img = img[y_min:y_max, x_min:x_max]
        cr_gray = cv2.cvtColor(cr_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(cr_gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for j in range(len(contours)):
            if cv2.contourArea(contours[j]) > 250:
                for k in range(len(contours[j])):
                    contours[j][k][0][0] = contours[j][k][0][0] + x_min
                    contours[j][k][0][1] = contours[j][k][0][1] + y_min
                cv2.drawContours(img, [contours[j]], 0, (0, 255, 0), 2)
            else:
                continue

    cv2.imshow('Contours', img)
    out.write(img)
    key = cv2.waitKey(1)
    if key == 27:
        out.release()
        break
