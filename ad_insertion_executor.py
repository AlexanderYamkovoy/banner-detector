from models.opencv_model.ad_insertion import AdInsertion
import cv2 as cv
import time
import numpy as np


def ad_insertion_executor(video_path, logo, config, file):
    """
    Execute AdInsertion model for logo insertion
    :param video_path: video path
    :param logo: logo path
    :param config: config path
    :param file: report path
    :return:
    """
    capture = cv.VideoCapture(video_path)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    four_cc = cv.VideoWriter_fourcc(*'FMP4')  # XVID FMP4 X264
    frames_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    file.write('Frames amount: {}\n'.format(frames_count))
    fps = capture.get(cv.CAP_PROP_FPS)
    file.write('FPS: {}\n'.format(int(fps)))
    out = cv.VideoWriter('result.avi', four_cc, fps, (frame_width, frame_height), True)

    # Preprocessing
    print('Preprocessing is running...')
    start_time = time.time()
    data = []
    for i in range(frames_count):
        _, frame = capture.read()
        ad_insertion = AdInsertion(frame, logo, i, data, None)
        ad_insertion.build_model(config)
        ad_insertion.data_preprocessed()
    data = np.array(data)
    np.save('data/data.npy', data)
    capture.release()
    end_time = time.time() - start_time
    file.write('Preprocessing runtime: {} seconds\n'.format(end_time))
    print('Preprocessing completed.')

    # Detection
    print('Detection is running...')
    start_time = time.time()
    ad_insertion = AdInsertion(None, None, None, None, fps)
    ad_insertion.build_model(config)
    contours_amount = ad_insertion.detect_surfaces()
    stable_contours = ad_insertion.stable_contours
    end_time = time.time() - start_time
    file.write('Detection runtime: {} seconds\n'.format(end_time))
    file.write('Stable contours amount: {}\n'.format(contours_amount))
    print('Detection completed.')

    # Ads insertion
    print('Insertion is running...')
    start_time = time.time()
    capture = cv.VideoCapture(video_path)
    for i in range(frames_count):
        _, frame = capture.read()
        if i in stable_contours[:, 0]:
            ad_insertion = AdInsertion(frame, logo, i, None, None)
            ad_insertion.build_model(config)
            ad_insertion.insert_ad(stable_contours)
        out.write(frame)
    end_time = time.time() - start_time
    file.write('Insertion runtime: {} seconds\n'.format(end_time))
    print('Insertion completed.')

    capture.release()
    out.release()


film_path = 'SET FILM NAME'
logo_path = 'SET LOGO NAME'
config_path = 'models/configurations/configurations.yml'

with open('data/report.txt', 'w') as report:
    total_start = time.time()
    ad_insertion_executor(film_path, logo_path, config_path, report)
    total_end = time.time() - total_start
    report.write('Total runtime: {} seconds\n'.format(total_end))
