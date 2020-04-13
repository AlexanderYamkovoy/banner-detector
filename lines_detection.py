import cv2 as cv
import numpy as np
import math
import pandas as pd
import time


def detect_lines(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # blur_gray = cv.GaussianBlur(gray, (3, 3), 0)

    _, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # th = cv.adaptiveThreshold(blur_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # laplacian = cv.Laplacian(gray, cv.CV_64F, ksize=3)
    # laplacian = np.uint8(np.absolute(laplacian))

    # sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    # sobely = np.uint8(np.absolute(sobely))

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
            if point[0][0] == max_x:
                max_y.append(point[0][1])

    max_y = np.min(max_y)

    base_line_tangent = (max_y - min_y)/(max_x - min_x)
    base_line_angle = math.degrees(math.atan(base_line_tangent))

    # for cnt in contours:
    #     cv.drawContours(image, [cnt], -1, (0, 255, 0), 2)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180
    threshold = 200  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200
    max_line_gap = 25  # maximum gap in pixels between connectible line segments
    line_image = np.copy(image) * 0

    cv.circle(line_image, (min_x, min_y), 5, (0, 0, 255), 2)
    cv.circle(line_image, (max_x, max_y), 5, (0, 0, 255), 2)

    cv.line(line_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    edges = cv.Canny(th, low_threshold, high_threshold)

    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:

            line_tangent = (y2 - y1) / (x2 - x1)
            line_angle = math.degrees(math.atan(line_tangent))

            # base_line_center = ((max_x - min_x) / 2, (max_y - min_y) / 2)

            # dst = abs((y2 - y1) * base_line_center[0] - (x2 - x1) * base_line_center[1] + x2 * y1 - y2 * x1) /\
                      # np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

            if abs(line_angle - base_line_angle) <= 0.15:
                color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                cv.line(line_image, (x1, y1), (x2, y2), color, 2)

    lines_edges = cv.addWeighted(image, 0.8, line_image, 1, 0)
    return lines_edges


def detect_shapes(image):

    kernel = 5
    min_area_threshold = 2000
    max_area_threshold = 80000
    perimeter_threshold = 0.035

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur_gray = cv.GaussianBlur(gray, (kernel, kernel), 0)

    _, th = cv.threshold(blur_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # th = cv.adaptiveThreshold(blur_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 2)
    # th = cv.adaptiveThreshold(blur_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    _, contours, __ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    drop_list = [i for i in range(len(contours))
                 if cv.contourArea(contours[i]) < min_area_threshold
                 or cv.contourArea(contours[i]) > max_area_threshold]
    contours = [i for j, i in enumerate(contours) if j not in drop_list]

    frames_contours = []
    for cnt in contours:
        epsilon = perimeter_threshold * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        convexity = cv.isContourConvex(approx)
        corners_count = len(approx)

        if convexity and corners_count == 4:
            frames_contours.append(approx.tolist())
            cv.drawContours(image, [approx], -1, (0, 255, 0), 2)
    return frames_contours


def data_collection(image):
    kernel = 5
    min_area_threshold = 2000
    max_area_threshold = 80000
    perimeter_threshold = 0.035

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur_gray = cv.GaussianBlur(gray, (kernel, kernel), 0)

    _, th = cv.threshold(blur_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    _, contours, __ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    drop_list = [i for i in range(len(contours))
                 if cv.contourArea(contours[i]) < min_area_threshold
                 or cv.contourArea(contours[i]) > max_area_threshold]
    contours = [i for j, i in enumerate(contours) if j not in drop_list]

    frames_contours = []
    for cnt in contours:
        epsilon = perimeter_threshold * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        convexity = cv.isContourConvex(approx)
        corners_count = len(approx)

        if convexity and corners_count == 4:
            frames_contours.append(approx.tolist())

    if len(frames_contours) > 0:
        return frames_contours
    else:
        return []


def data_preprocessed(contours, frame_idx, df, row_index):

    if contours != 0:
        for i, v in enumerate(contours):
            df.loc[row_index] = frame_idx, \
                                v[0][0][0], v[0][0][1], v[1][0][0], v[1][0][1], \
                                v[2][0][0], v[2][0][1], v[3][0][0], v[3][0][1]
            row_index += 1

    return row_index


def data_cleaning(df, min_frames_quantity, file):
    data = pd.read_csv(df)

    frames_idx = []
    j = 0
    for idx in range(len(data.frame) - 1):
        if data.frame[idx+1] - data.frame[idx] == 1:
            j += 1
        elif data.frame[idx+1] - data.frame[idx] == 0:
            continue
        else:
            j = 0
        frames_idx.append([data.frame[idx], j])

    idx_counter = []
    for item in frames_idx:
        if item[1] >= min_frames_quantity:
            idx_counter.append(item)

    final_frames = []
    for i in range(len(idx_counter)):
        if idx_counter[i][1] == min_frames_quantity:
            if i != 0:
                final_frames.append([(idx_counter[i-1][0] + 1) - idx_counter[i-1][1],
                                     idx_counter[i-1][0] + 1])
        if i == len(idx_counter) - 1:
            final_frames.append([(idx_counter[i - 1][0] + 1) - idx_counter[i - 1][1],
                                 idx_counter[i - 1][0] + 2])

    fragments = []
    for fragment in final_frames:
        file.write('Captured fragment: {}\n'.format(fragment))
        fragments.append(np.arange(fragment[0], fragment[1] + 1))

    idx_list = []
    for fragment in fragments:
        for idx in fragment:
            idx_list.append(idx)

    return idx_list


def drawing_contours(csv, idx_frame, frame):
    data = pd.read_csv(csv)

    for i, _ in data.iterrows():
        if data.frame[i] == idx_frame:
            contour = [[[data.x1[i], data.y1[i]]], [[data.x2[i], data.y2[i]]],
                       [[data.x3[i], data.y3[i]]], [[data.x4[i], data.y4[i]]]]
            contour = np.array(contour)
            cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)


def execution(video, img_path, video_path='Set path', preprocessing=True):
    if not video:
        img = cv.imread(img_path)
        cnt = detect_shapes(img)
        cv.imshow('result', img)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()
    else:
        capture = cv.VideoCapture(video_path)
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        four_cc = cv.VideoWriter_fourcc(*'FMP4')  # XVID FMP4 X264
        frames_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv.CAP_PROP_FPS)
        out = cv.VideoWriter('/home/worker/Shape_Detector/results/draw_quadrilaterals.avi',
                             four_cc, fps, (frame_width, frame_height), True)
        if preprocessing:
            columns = ['frame', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
            data = pd.DataFrame(columns=columns)
            row_idx = 0

            for i in range(frames_count):
                _, frame = capture.read()
                contours = data_collection(frame)
                row_idx = data_preprocessed(contours, i, data, row_idx)
            data.to_csv('data.csv', index=False)

        with open('report.txt', 'w') as f:
            f.write('Frames count: {}\n'.format(frames_count))
            f.write('FPS amount: {}\n'.format(fps))
            frames_ranges = data_cleaning('data.csv', 90, f)

        for i in range(frames_count):
            _, frame = capture.read()
            if i in frames_ranges:
                drawing_contours('data.csv', i, frame)

            # cv.imshow('frame', frame)
            out.write(frame)

            # key = cv.waitKey(1)
            # if key == 27:
            #     break

        capture.release()
        out.release()


folder = '/home/worker/Shape_Detector/Avengers.mkv'
image_name = 'frame434.png'

start_time = time.time()
execution(True, image_name, folder)
print("--- %s seconds ---" % (time.time() - start_time))
