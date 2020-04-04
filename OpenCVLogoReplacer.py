import numpy as np
import pandas as pd
import cv2 as cv
from LogoReplacer import LogoReplacer
import yaml
import math


class OpenCVLogoReplacer(LogoReplacer):
    def __init__(self, input_frame, logo_path):
        self.frame = input_frame
        self.logo = logo_path
        self.parameters = {}
        self.corners = 0
        self.frame_num = 0

    def __field_detection(self, kp_template, matcher, min_match_count, dst_threshold, n_features, rc_threshold):
        gray_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        template = cv.imread(kp_template)
        gray_template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

        sift = cv.xfeatures2d.SIFT_create(n_features)

        kp1, des1 = sift.detectAndCompute(gray_template, None)
        kp2, des2 = sift.detectAndCompute(gray_frame, None)

        index_params = {'algorithm': matcher['index_params'][0], 'trees': matcher['index_params'][1]}
        search_params = {'checks': matcher['search_params']}
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < dst_threshold * n.distance:
                good.append(m)

        field = None
        if len(good) >= min_match_count:

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            m, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, rc_threshold)
            h, w = gray_template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, m)

            for i in range(len(dst)):
                if dst[i][0][1] < 0:
                    dst[i][0][1] = 0

            # cv.polylines(self.frame, [np.int32(dst)], True, 255, 2, cv.LINE_AA)
            x_corner_list = [dst[i][0][0] for i in range(len(dst))]
            y_corner_list = [dst[j][0][1] for j in range(len(dst))]
            x_min, x_max = np.int64(min(x_corner_list)), np.int64(max(x_corner_list))
            y_min, y_max = np.int64(min(y_corner_list)), np.int64(max(y_corner_list))
            field = self.frame[y_min:y_max, x_min:x_max]
            return field, [x_min, y_min]
        else:
            return field, 0

    def __shape_detection(self, img, origin, kernel, area_threshold):
        h, w, _ = self.frame.shape
        frame_copy = self.frame.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (kernel, kernel), 0)
        _, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        _, contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        '''small_areas = [i for i in range(len(contours)) if
                       (cv.contourArea(contours[i]) < 1000)]
        contours = [i for j, i in enumerate(contours) if j not in small_areas]

        filtered_cnt = []
        for cnt in contours:
            m = cv.moments(cnt)
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            bias_x = abs(cx - w/2)
            bias_y = abs(cy - h/2)

            if bias_x < 220 and cx < 640 and bias_y < 150 and cv.contourArea(cnt) < 40000:
                max_y_idx = np.ravel(np.argmax(cnt, axis=0))[1]
                min_x_idx = np.ravel(np.argmin(cnt, axis=0))[0]
                if cnt[max_y_idx][0][1] in range(400, 600) and cnt[min_x_idx][0][0] in range(300, 500):
                    epsilon = 0.01 * cv.arcLength(cnt, True)
                    approx = cv.approxPolyDP(cnt, epsilon, True)
                    cv.drawContours(self.frame, [approx], -1, (0, 255, 0), 3)'''

        # SMALL POSTER(2)
        '''area_list = [cv.contourArea(cnt) for cnt in contours]
        max_area_index = area_list.index(max(area_list))
        max_contour = contours[max_area_index]

        for i in range(len(max_contour)):
            max_contour[i][0][0] = max_contour[i][0][0] + origin[0]
            max_contour[i][0][1] = max_contour[i][0][1] + origin[1]

        minx_index, miny_index = np.ravel(np.argmin(max_contour, axis=0))
        min_x = max_contour[minx_index][0][0]
        min_y = max_contour[miny_index][0][1]
        final = []
        for i in range(len(max_contour)):
            approx_h = abs(min_y - max_contour[i][0][1])
            approx_w = abs(min_x - max_contour[i][0][0])
            if approx_h < 212 and approx_w < 334:
                final.append(max_contour[i])
        final = np.array(final)

        epsilon = 0.055 * cv.arcLength(final, True)
        approx = cv.approxPolyDP(final, epsilon, True)
        approx2 = []
        for i in range(len(approx)):
            if approx[i][0][0] in range(225, 235) or approx[i][0][0] in range(500, 565):
                approx2.append(approx[i])
        approx2 = np.array(approx2)

        cv.drawContours(frame_copy, [approx2], -1, (0, 255, 0), -1)

        index_max_list = np.ravel(np.argmax(approx2, axis=0))
        index_min_list = np.ravel(np.argmin(approx2, axis=0))

        top_left_x = approx2[index_min_list[0]][0][0]
        top_left_y = approx2[index_min_list[1]][0][1]
        bot_left_x = top_left_x
        bot_left_y = approx2[index_max_list[1]][0][1]

        top_left = [top_left_x, top_left_y]
        bot_left = [bot_left_x, bot_left_y]
        bot_right = [top_left_x + 330, bot_left_y]
        top_right = [top_left_x + 330, top_left_y]
        # self.corners = [top_left, bot_left, bot_right, top_right]
        self.corners = corners'''

        # SMALL POSTER (1)
        '''m = cv.moments(th)
        cx = int(m['m10'] / m['m00']) + origin[0]
        cy = int(m['m01'] / m['m00']) + origin[1]

        small_areas = [i for i in range(len(contours)) if
                       (cv.contourArea(contours[i]) < 7000)]
        contours = [i for j, i in enumerate(contours) if j not in small_areas]

        distances = []
        for cnt in contours:
            for i in range(len(cnt)):
                cnt[i][0][0] = cnt[i][0][0] + origin[0]
                cnt[i][0][1] = cnt[i][0][1] + origin[1]
            index_min_list = np.ravel(np.argmin(cnt, axis=0))
            top_x = cnt[index_min_list[0]][0][0]
            top_y = cnt[index_min_list[1]][0][1]
            distance = math.sqrt((top_x - cx)**2 + (top_y - cy)**2)
            distances.append(distance)

        min_distance = distances.index(min(distances))
        contour = contours[min_distance]

        minx_index, miny_index = np.ravel(np.argmin(contour, axis=0))
        contour_minx = contour[minx_index][0][0]
        contour_miny = contour[miny_index][0][1]
        final_contour = []
        for i in range(len(contour)):
            approx_h = abs(contour[i][0][0] - contour_minx)
            approx_w = abs(contour[i][0][1] - contour_miny)
            if (approx_h < 132) and (approx_w < 88):
                final_contour.append(contour[i])
        final_contour = np.array(final_contour)
        cv.drawContours(frame_copy, [final_contour], -1, (0, 255, 0), -1)

        index_max_list = np.ravel(np.argmax(final_contour, axis=0))
        index_min_list = np.ravel(np.argmin(final_contour, axis=0))
        
        top_left_x = final_contour[index_min_list[0]][0][0]
        top_left_y = final_contour[index_min_list[1]][0][1]
        bot_right_x = final_contour[index_max_list[0]][0][0]
        bot_right_y = final_contour[index_max_list[1]][0][1]
        
        top_left = [top_left_x, top_left_y]
        bot_left = [top_left_x, bot_right_y]
        bot_right = [bot_right_x, bot_right_y]
        top_right = [bot_right_x, top_left_y]

        # self.corners = [top_left, bot_left, bot_right, top_right]
        self.corners = corners'''

        # POSTER 2
        area_list = [cv.contourArea(cnt) for cnt in contours]
        max_area_index = area_list.index(max(area_list))
        required_contour = contours[max_area_index]
        for i in range(len(required_contour)):
            required_contour[i][0][0] = required_contour[i][0][0] + origin[0]
            required_contour[i][0][1] = required_contour[i][0][1] + origin[1]

        epsilon = area_threshold * cv.arcLength(required_contour, True)
        approx = cv.approxPolyDP(required_contour, epsilon, True)
        cv.drawContours(frame_copy, [approx], -1, (0, 255, 0), -1)

        index_max_list = np.ravel(np.argmax(approx, axis=0))
        index_min_list = np.ravel(np.argmin(approx, axis=0))

        '''top_left = approx[index_min_list[1]].tolist()[0]
        bot_left = approx[index_min_list[0]].tolist()[0]

        x_max = approx[np.where(approx == frame_copy.shape[1] - 1)[0].tolist()]
        new_index_max = np.ravel(np.argmax(x_max, axis=0))[1]
        new_index_min = np.ravel(np.argmin(x_max, axis=0))[1]
        bot_right = x_max[new_index_max].tolist()[0]
        top_right = x_max[new_index_min].tolist()[0]'''

        top_left = approx[index_min_list[0]].tolist()[0]
        bot_left = approx[index_max_list[1]].tolist()[0]
        bot_right = approx[index_max_list[0]].tolist()[0]
        top_right = approx[index_min_list[1]].tolist()[0]

        # transform part of poster
        '''if self.frame_num < 1750:

            left_height = abs(top_left[1] - bot_left[1])
            right_height = abs(top_right[1] - bot_right[1])
            top_width = abs(top_left[0] - top_right[0])
            bot_width = abs(bot_left[0] - bot_right[0])

            ratio_top = left_height / top_width
            ratio_bot = left_height / bot_width

            if bot_right[0] >= 1276 or top_right[0] >= 1276:
                y_top = lambda x: (x - top_left[0]) * (top_right[1] - top_left[1]) / (top_right[0] - top_left[0]) + \
                                  top_left[1]
                y_bot = lambda x: (x - bot_left[0]) * (bot_right[1] - bot_left[1]) / (bot_right[0] - bot_left[0]) + \
                                  bot_left[1]

                if right_height > left_height:
                    bot_left[1] = top_left[1] + right_height
                else:
                    bot_right[1] = top_right[1] + left_height

                if top_width > bot_width:
                    tmp_bot_r_x = top_left[0] + top_width
                    bot_right[1] = y_bot(tmp_bot_r_x)
                    bot_right[0] = tmp_bot_r_x
                else:
                    tmp_top_r_x = top_left[0] + bot_width
                    top_right[1] = y_top(tmp_top_r_x)
                    top_right[0] = tmp_top_r_x

                ratio_top = left_height / top_width
                ratio_bot = left_height / bot_width

                # if abs(ratio_top - 1.27) > 0.05 or abs(ratio_bot - 1.22) > 0.05:
                bot_left[1] = top_left[1] + top_width * (ratio_top + 0.25)
                left_height = abs(top_left[1] - bot_left[1])
                tmp_top_r_x = top_left[0] + left_height / ratio_top
                tmp_bot_r_x = bot_left[0] + left_height / ratio_bot

                top_right[1] = y_top(tmp_top_r_x)
                bot_right[1] = y_bot(tmp_bot_r_x)
                top_right[0] = tmp_top_r_x
                bot_right[0] = tmp_bot_r_x

        if self.frame_num > 1749:

            left_height = np.sqrt((top_left[0] - bot_left[0]) ** 2 + (top_left[1] - bot_left[1]) ** 2)
            top_width = abs(top_left[0] - top_right[0])
            bot_width = abs(bot_left[0] - bot_right[0])
            ratio_top = left_height / top_width
            ratio_bot = left_height / bot_width

            if abs(ratio_top - 1.26) > 0.1 or abs(ratio_bot - 1.26) > 0.1:
                tmp_top_r_x = top_left[0] + left_height / 1.27
                tmp_bot_r_x = bot_left[0] + left_height / 1.22
                y_top = lambda x: (x - top_left[0]) * (top_right[1] - top_left[1]) / (top_right[0] - top_left[0]) + \
                                  top_left[1]

                y_bot = lambda x: (x - bot_left[0]) * (bot_right[1] - bot_left[1]) / (bot_right[0] - bot_left[0]) + \
                                  bot_left[1]

                top_right[1] = y_top(tmp_top_r_x)
                bot_right[1] = y_bot(tmp_bot_r_x)
                top_right[0] = tmp_top_r_x
                bot_right[0] = tmp_bot_r_x
        # self.corners = [top_left, bot_left, bot_right, top_right]
        self.corners = corners'''

        # transform big poster
        if bot_right[0] == top_right[0]:
            new_array = approx[np.where(approx == frame_copy.shape[1] - 1)[0].tolist()]
            new_index_max = np.ravel(np.argmax(new_array, axis=0))
            bot_right = new_array[new_index_max[1]].tolist()[0]

        left_height = np.sqrt((top_left[0] - bot_left[0]) ** 2 + (top_left[1] - bot_left[1]) ** 2)
        top_width = np.sqrt((top_left[0] - top_right[0]) ** 2)
        bot_width = np.sqrt((bot_left[0] - bot_right[0]) ** 2)
        ratio_top = left_height / top_width
        ratio_bot = left_height / bot_width

        if abs(ratio_top - 1.26) > 0.1 or abs(ratio_bot - 1.26) > 0.1:
            tmp_top_r_x = top_left[0] + left_height / 1.27
            tmp_bot_r_x = bot_left[0] + left_height / 1.22
            y_top = lambda x: (x - top_left[0]) * (top_right[1] - top_left[1]) / (top_right[0] - top_left[0]) + \
                              top_left[1]

            y_bot = lambda x: (x - bot_left[0]) * (bot_right[1] - bot_left[1]) / (bot_right[0] - bot_left[0]) + \
                              bot_left[1]

            top_right[1] = y_top(tmp_top_r_x)
            bot_right[1] = y_bot(tmp_bot_r_x)
            top_right[0] = tmp_top_r_x
            bot_right[0] = tmp_bot_r_x

        self.corners = [top_left, bot_left, bot_right, top_right]
        # self.corners = corners

        return frame_copy

    def __transform_logo(self):
        self.logo = cv.imread(self.logo)

        frame_h, frame_w, _ = self.frame.shape
        h, w, _ = self.logo.shape

        pts1 = np.float32([(0, 0), (0, (h - 1)), ((w - 1), (h - 1)), ((w - 1), 0)])
        pts2 = np.float32([self.corners[0], self.corners[1], self.corners[2], self.corners[3]])

        matrix = cv.getPerspectiveTransform(pts1, pts2)
        self.logo = cv.warpPerspective(self.logo, matrix, (frame_w, frame_h), borderMode=1)

    def build_model(self, filename):
        with open(filename, 'r') as stream:
            self.parameters = yaml.safe_load(stream)

    def detect_object(self):
        p = self.parameters
        field, origin = self.__field_detection(p['kp_template'], p['matcher'], p['min_match_count'],
                                               p['dst_threshold'], p['n_features'], p['rc_threshold'])
        if field is not None:
            frame_copy = self.__shape_detection(field, origin, p['kernel'], p['area_threshold'])
            self.__transform_logo()
            return frame_copy, self.corners
        else:
            return None, 0

    def insert_logo(self, frame_copy):
        # frame_copy = cv.imread(''.format(i))
        for i in range(self.frame.shape[0]):
            for j in range(self.frame.shape[1]):
                if list(frame_copy[i, j]) == [0, 255, 0]:
                    self.frame[i, j] = self.logo[i, j]


if __name__ == '__main__':
    logo = ''
    video = ''
    image = ''
    write_video = True

    if write_video:
        capture = cv.VideoCapture(video)
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        fps = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        four_cc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter('video.avi', four_cc, 30, (frame_width, frame_height), True)
        # smoothed_data = pd.read_csv('')
        for i in range(fps):
            _, frame = capture.read()
            # if i in smoothed_data['frame'].values:
            logo_replacer = OpenCVLogoReplacer(frame, logo)
            logo_replacer.build_model('parameters_setting.yml')
            detected, _ = logo_replacer.detect_object()

            '''top_left = [smoothed_data[smoothed_data['frame'] == i].values[0][1],
                            smoothed_data[smoothed_data['frame'] == i].values[0][2]]
            bot_left = [smoothed_data[smoothed_data['frame'] == i].values[0][3],
                            smoothed_data[smoothed_data['frame'] == i].values[0][4]]
            bot_right = [smoothed_data[smoothed_data['frame'] == i].values[0][5],
                             smoothed_data[smoothed_data['frame'] == i].values[0][6]]
            top_right = [smoothed_data[smoothed_data['frame'] == i].values[0][7],
                             smoothed_data[smoothed_data['frame'] == i].values[0][8]]

            corners = [top_left, bot_left, bot_right, top_right]'''

            if detected is not None:
                logo_replacer.insert_logo(detected)

            cv.imshow('video', frame)
            out.write(frame)
            key = cv.waitKey(1)
            if key == 27:
                break
        capture.release()
        out.release()
    else:
        image = cv.imread(image)
        logo_replacer = OpenCVLogoReplacer(image, logo)
        logo_replacer.build_model('parameters_setting.yml')
        detected = logo_replacer.detect_object()
        # if detected is not None:
        #     logo_replacer.insert_logo(detected)
        cv.imshow('image', image)
        key = cv.waitKey()
        if key == 27:
            cv.destroyAllWindows()
