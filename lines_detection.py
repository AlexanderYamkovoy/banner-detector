import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt


def detect_lines(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    laplacian = cv.Laplacian(gray, cv.CV_64F, ksize=3)
    laplacian = np.uint8(np.absolute(laplacian))

    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    sobely = np.uint8(np.absolute(sobely))

    low_threshold = 30
    high_threshold = 60

    low_color = np.array([35, 85, 60])  # 35 25 25
    high_color = np.array([50, 255, 255])  # 70 255 255
    color_mask = cv.inRange(hsv, low_color, high_color)
    _, contours, __ = cv.findContours(color_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    drop_list = [i for i in range(len(contours)) if cv.contourArea(contours[i]) < 2500]
    contours = [i for j, i in enumerate(contours) if j not in drop_list]

    min_y = np.min([[np.min([point[0][1] for point in contours[i]])] for i in range(len(contours))])

    endpoints = []
    for cnt in contours:
        for point in cnt:
            if point[0][1] == min_y:
                endpoints.append(point[0])

    min_x = np.min([point[0] for point in endpoints])
    max_x = image.shape[1] - 1

    max_y = []
    for i in range(len(contours)):
        for point in contours[i]:
            if point[0][0] == image.shape[1] - 1:
                max_y.append(point[0][1])

    max_y = np.min(max_y)

    cv.circle(image, (min_x, min_y), 5, (0, 0, 255), 2)
    cv.circle(image, (max_x, max_y), 5, (0, 0, 255), 2)

    cv.line(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    # for cnt in contours:
    #     cv.drawContours(image, [cnt], -1, (0, 255, 0), 2)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180
    threshold = 180  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 150
    max_line_gap = 25  # maximum gap in pixels between connectible line segments
    line_image = np.copy(image) * 0

    edges = cv.Canny(gray, low_threshold, high_threshold)

    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if y1 & y2 in range(min_y - 15, min_y + 50):
                color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                cv.line(line_image, (x1, y1), (x2, y2), color, 2)

    lines_edges = cv.addWeighted(image, 0.8, line_image, 1, 0)
    return image


def build_model(video, img_path='Set path', files_folder_path='Set path'):
    if not video:
        img = cv.imread(img_path)
        lines = detect_lines(img)
        cv.imshow('result', lines)
        cv.imwrite(files_folder_path + 'single_line.png', lines)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()
    else:
        capture = cv.VideoCapture(files_folder_path + 'football.mp4')
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        four_cc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter(files_folder_path + 'single_line.avi', four_cc, 30, (frame_width, frame_height), True)
        while capture.isOpened():
            res, frame = capture.read()
            if res:
                lines = detect_lines(frame)
                cv.imshow('result', lines)
                out.write(lines)
                key = cv.waitKey(1)
                if key == 27:
                    break
            else:
                break

        capture.release()
        out.release()


folder = '/Users/oleksandr/Folder/WinStars/'
build_model(False, folder + 'frame10.png', folder)
