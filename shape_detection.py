import numpy as np
import cv2 as cv


def shape_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # _, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    '''corners = cv.goodFeaturesToTrack(th, 30, 0.01, 10)
    for corner in corners:
        cv.circle(image, (corner[0][0], corner[0][1]), 5, (0, 0, 255))'''

    '''gray = np.float32(gray)
    dst = cv.cornerHarris(th, 2, 3, 0.04)
    
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    image[dst > 0.01*dst.max()] = [0, 0, 255]'''

    _, contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    drop_list = [i for i in range(len(contours)) if cv.contourArea(contours[i]) < 3000]
    contours = [i for j, i in enumerate(contours) if j not in drop_list]

    for cnt in contours:
        cv.drawContours(image, [cnt], -1, (0, 255, 0), 2)


def save_result(video):
    capture = cv.VideoCapture(video)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    four_cc = cv.VideoWriter_fourcc(*'MJPG')
    # out = cv.VideoWriter('shape_detection2.avi', four_cc, 30, (frame_width, frame_height), True)
    while capture.isOpened():
        res, frame = capture.read()
        if res:
            shape_detection(frame)
            cv.imshow('result', frame)
            # out.write(frame)
            key = cv.waitKey(1)
            if key == 27:
                break
        else:
            break

    capture.release()
    # out.release()


def switch(x):
    method = {'GoodFeaturesToTrack': 1,
              'CornerHarris': 2,
              'FindContours': 3}
    return method[x]


save_result('/Users/oleksandr/Folder/superman/test.mp4')
