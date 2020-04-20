import cv2 as cv
import numpy as np
import math
import pandas as pd
import time
from skimage.exposure import match_histograms
from AbstractAdInsertion import AbstractAdInsertion


class AdInsertion(AbstractAdInsertion):
    def __init__(self, frame, logo, data, row_idx, frame_idx):
        self.frame = frame
        self.logo = logo
        self.contours = []
        self.data = data
        self.row_idx = row_idx
        self.frame_idx = frame_idx

    def contours_finding(self):
        kernel = 5
        min_area_threshold = 2000
        max_area_threshold = 80000
        perimeter_threshold = 0.035
        corners_count = 4
        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        blur_gray = cv.GaussianBlur(gray, (kernel, kernel), 0)

        _, th = cv.threshold(blur_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, contours, __ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        drop_list = [i for i in range(len(contours))
                     if cv.contourArea(contours[i]) < min_area_threshold
                     or cv.contourArea(contours[i]) > max_area_threshold]
        contours = [i for j, i in enumerate(contours) if j not in drop_list]

        for cnt in contours:
            epsilon = perimeter_threshold * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)
            corners = len(approx)
            convexity = cv.isContourConvex(approx)

            if convexity and corners == corners_count:
                self.contours.append(approx.tolist())

    def data_preprocessed(self):

        if self.contours != 0:
            for i, v in enumerate(self.contours):
                self.data.loc[self.row_idx] = self.frame_idx, \
                                    v[0][0][0], v[0][0][1], v[1][0][0], v[1][0][1], \
                                    v[2][0][0], v[2][0][1], v[3][0][0], v[3][0][1]
                self.row_index += 1

    def data_cleaning(self, df, file):
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
            if item[1] >= 60:
                idx_counter.append(item)

        final_frames = []
        for i in range(len(idx_counter)):
            if idx_counter[i][1] == 60:
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
