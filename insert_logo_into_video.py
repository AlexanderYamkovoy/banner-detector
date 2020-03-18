import cv2 as cv
import numpy as np
from OpenCVLogoInsertion import OpenCVLogoInsertion
from banner_parameters_setting import banner_parameters_setting
import pandas as pd


def frames_capture(video, show_details=False):
    """
    The function provides all frames from the input video

    :param video: input video file
    :param show_details: true if you need to know total amount of frames and fps
    :return:
    """
    capture = cv.VideoCapture(video)
    frames_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv.CAP_PROP_FPS)

    if show_details:
        print('Total amount of frames:', frames_count)
        print('FPS amount:', fps)

    frames_list = []

    while capture.isOpened():
        res, image = capture.read()
        if res:
            frames_list.append(image)
        else:
            capture.release()
    for i in range(0, frames_count):
        cv.imwrite('SET PATH FOR PASTING FRAMES'.format(i), frames_list[i])


def smoothing_corners(filename, i):
    """
    This function return smoothed corners coordinates for corresponding frame
    :param filename: file with smoothed corners
    :param i: frame index
    :return: smoothed corners coordinates
    """
    data = pd.read_csv(filename, delimiter=';')
    top_left = [data.iloc[i, 0], data.iloc[i, 1]]
    bot_left = [data.iloc[i, 4], data.iloc[i, 5]]
    bot_right = [data.iloc[i, 6], data.iloc[i, 7]]
    top_right = [data.iloc[i, 2], data.iloc[i, 3]]
    true_corners = [top_left, bot_left, bot_right, top_right]
    return true_corners


def insert_logo_into_video(video, write_video=True, show_f1=False):
    """
    This function provides logo insertion into the video file

    :param video: input video file
    :param write_video: choose True if you want to write the video
    :param show_f1: True if need to calculate f1 score
    :return: mean value for f1 score
    """
    additional_templates = ['SET ADDITIONAL TEMPLATE NAME']
    banner_parameters_setting()

    capture = cv.VideoCapture(video)
    frames_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    four_cc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('RESULTING VIDEO NAME', four_cc, 30, (frame_width, frame_height), True)

    n = 0
    f1_list = []
    for i in range(frames_count - 1):
        true_corners = smoothing_corners('SET PREPROCESSED CORNERS FILE PATH', i)
        ret, frame = capture.read()

        if ret:
            # frame_n = 'frame{}.jpg'.format(n)
            open_cv_insertion = OpenCVLogoInsertion('SET TEMPLATE', frame, 'SET LOGO')
            open_cv_insertion.build_model('SET PARAMETERS')
            cropped_frame, resized_banner, frame_copy, switch_, f1\
                = open_cv_insertion.detect_banner(None, true_corners)

            if switch_:
                open_cv_insertion.insert_logo(resized_banner, frame_copy)

            else:
                for tmp in additional_templates:
                    open_cv_insertion = OpenCVLogoInsertion(tmp, frame, 'SET LOGO')
                    open_cv_insertion.build_model('SET PARAMETERS')
                    cropped_frame, resized_banner, frame_copy, switch_, f1\
                        = open_cv_insertion.detect_banner(None, true_corners)

                    if switch_:
                        open_cv_insertion.insert_logo(resized_banner, frame_copy)
                        break
                    else:
                        continue

            if f1 != -1 and show_f1:
                f1_list.append(f1)

            if write_video:
                out.write(frame)

            key = cv.waitKey(1)
            if key == 27:
                break
            n += 1
        else:
            break
    capture.release()
    out.release()
    if len(f1_list) > 0:
        return np.mean(f1_list)
    else:
        pass


# frames_capture('SET VIDEO NAME')

insert_logo_into_video('SET VIDEO NAME')
