import cv2 as cv
import numpy as np
import math
import pandas as pd
import time
from skimage.exposure import match_histograms


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

    # fragments_list = [1, 2, 9, 15, 40, 48, 70, 73, 74, 76, 79, 94, 111, 117, 121]
    # res_fragments = []
    # for idx, fragment in enumerate(fragments):
    #     if idx in fragments_list:
    #         res_fragments.append(fragment)

    idx_list = []
    for fragment in fragments:
        for idx in fragment:
            idx_list.append(idx)

    return idx_list


def drawing_contours(csv, idx_frame, frame):
    data = pd.read_csv(csv)

    required_data = data[data.frame == idx_frame]
    cnt_corners = []
    cnt_min_max = []
    for i, _ in required_data.iterrows():
        contour = np.array([[[required_data.x1[i], required_data.y1[i]]],
                           [[required_data.x2[i], required_data.y2[i]]],
                           [[required_data.x3[i], required_data.y3[i]]],
                           [[required_data.x4[i], required_data.y4[i]]]])

        x_min, y_min = np.amin(contour, axis=0)[0], np.amin(contour, axis=0)[1]
        x_max, y_max = np.amax(contour, axis=0)[0], np.amax(contour, axis=0)[1]
        cnt_min_max.append([y_min, y_max, x_min, x_max])
        contour.view('i8,i8').sort(order=['f0'], axis=0)

        left_side = contour[:2]
        right_side = contour[2:]

        left_idx_max = np.ravel(np.argmax(left_side, axis=0))[1]
        left_idx_min = np.ravel(np.argmin(left_side, axis=0))[1]
        right_idx_max = np.ravel(np.argmax(right_side, axis=0))[1]
        right_idx_min = np.ravel(np.argmin(right_side, axis=0))[1]

        top_left = left_side[left_idx_min].tolist()[0]
        bot_left = left_side[left_idx_max].tolist()[0]
        top_right = right_side[right_idx_min].tolist()[0]
        bot_right = right_side[right_idx_max].tolist()[0]
        cnt_corners.append([[top_left], [bot_left], [bot_right], [top_right]])

        cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    return cnt_corners, cnt_min_max


def transform_logo(logo, frame, corners, min_max):
    logo = cv.imread(logo)

    frame_h, frame_w, _ = frame.shape
    h, w, _ = logo.shape

    pts1 = np.float32([(0, 0), (0, (h - 1)), ((w - 1), (h - 1)), ((w - 1), 0)])
    pts2 = np.float32([corners[0], corners[1], corners[2], corners[3]])

    matrix = cv.getPerspectiveTransform(pts1, pts2)
    logo = cv.warpPerspective(logo, matrix, (frame_w, frame_h), borderMode=1)

    matched = match_histograms(logo[min_max[0]:min_max[1], min_max[2]:min_max[3]],
                               frame[min_max[0]:min_max[1], min_max[2]:min_max[3]], multichannel=True)
    logo[min_max[0]:min_max[1], min_max[2]:min_max[3]] = matched

    return logo


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
        # out = cv.VideoWriter('draw_quadrilaterals_fragments.avi',
        #                      four_cc, fps, (frame_width, frame_height), True)
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
            frames_ranges = data_cleaning('data.csv', 60, f)

        for i in range(frames_count):
            _, frame = capture.read()
            if i in frames_ranges:
                print('Processing frame {}'.format(i))
                drawing_contours('data.csv', i, frame)
                # out.write(frame)

            # cv.imshow('frame', frame)
            # out.write(frame)

            # key = cv.waitKey(1)
            # if key == 27:
            #     break

        capture.release()
        # out.release()


folder = '/home/worker/Shape_Detector/Avengers.mkv'
local_video_path = '/Users/oleksandr/Folder/WinStars/avengers.mp4'
image_name = 'frame434.png'

start_time = time.time()
execution(True, image_name, local_video_path, False)
print("--- %s seconds ---" % (time.time() - start_time))

