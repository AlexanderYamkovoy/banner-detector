import numpy as np
import cv2 as cv


def shape_detection(image, origin, frame):
    if image is not None:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)

        _, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        # th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

        _, contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # drop_list = [i for i in range(len(contours)) if cv.contourArea(contours[i]) < 3500]
        # contours = [i for j, i in enumerate(contours) if j not in drop_list]

        area_list = [cv.contourArea(cnt) for cnt in contours]
        max_area_index = area_list.index(max(area_list))
        required_contour = contours[max_area_index]
        for i in range(len(required_contour)):
            required_contour[i][0][0] = required_contour[i][0][0] + origin[0]
            required_contour[i][0][1] = required_contour[i][0][1] + origin[1]

        epsilon = 0.01 * cv.arcLength(required_contour, True)
        approx = cv.approxPolyDP(required_contour, epsilon, True)
        # cv.drawContours(frame, [approx], -1, (0, 255, 0), 2)

        index_max_list = np.ravel(np.argmax(approx, axis=0))
        index_min_list = np.ravel(np.argmin(approx, axis=0))

        top_left = approx[index_min_list[0]].tolist()[0]
        bot_left = approx[index_max_list[1]].tolist()[0]
        bot_right = approx[index_max_list[0]].tolist()[0]
        top_right = approx[index_min_list[1]].tolist()[0]

        '''if bot_right[0] == top_right[0]:
            new_array = approx[np.where(approx == 1279)[0].tolist()]
            new_index_max = np.ravel(np.argmax(new_array, axis=0))
            bot_right = new_array[new_index_max[1]].tolist()[0]'''
        corners = [top_left, bot_left, bot_right, top_right]
        pts = np.array(corners, np.int32)
        cv.fillPoly(frame, [pts], (0, 0, 255), lineType=cv.LINE_AA)

    else:
        pass


def field_detection(frame, template):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    template = cv.imread(template)
    gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SIFT_create(10000)

    kp1, des1 = sift.detectAndCompute(gray_template, None)
    kp2, des2 = sift.detectAndCompute(gray, None)

    index_params = {'algorithm': 1, 'trees': 5}
    search_params = {'checks': 70}
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) >= 20:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        h, w = gray_template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, m)
        for i in range(len(dst)):
            if dst[i][0][1] < 0:
                dst[i][0][1] = 0
        # cv.polylines(frame, [np.int32(dst)], True, 255, 2, cv.LINE_AA)
        x_corner_list = [dst[i][0][0] for i in range(len(dst))]
        y_corner_list = [dst[j][0][1] for j in range(len(dst))]
        x_min, x_max = np.int64(min(x_corner_list)), np.int64(max(x_corner_list))
        y_min, y_max = np.int64(min(y_corner_list)), np.int64(max(y_corner_list))
        cr_frame = frame[y_min:y_max, x_min:x_max]
        return cr_frame, [x_min, y_min]
    else:
        return None, 0


def save_result(video, tmp, single=False):
    if single:
        image = cv.imread('/Users/oleksandr/Folder/superman/frame2210.png')
        cr, ori = field_detection(image, tmp)
        shape_detection(cr, ori, image)
        cv.imshow('result', image)
        key = cv.waitKey()
        if key == 27:
            cv.destroyAllWindows()
    else:
        capture = cv.VideoCapture(video)
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        four_cc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter('field_detection33.avi', four_cc, 30, (frame_width, frame_height), True)
        while capture.isOpened():
            res, frame = capture.read()
            if res:
                cr, ori = field_detection(frame, tmp)
                shape_detection(cr, ori, frame)
                cv.imshow('result', frame)
                out.write(frame)
                key = cv.waitKey(1)
                if key == 27:
                    break
            else:
                break

        capture.release()
        out.release()


def switch(x):
    method = {'GoodFeaturesToTrack': 1,
              'CornerHarris': 2,
              'FindContours': 3}
    return method[x]


save_result('/Users/oleksandr/Folder/superman/test.mp4', '/Users/oleksandr/Folder/superman/ff.png')
