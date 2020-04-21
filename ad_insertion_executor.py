from ad_insertion import AdInsertion
import cv2 as cv
import time
import pandas as pd
import numpy as np


def ad_insertion_executor(video_path, logo, config):
    capture = cv.VideoCapture(video_path)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    four_cc = cv.VideoWriter_fourcc(*'FMP4')  # XVID FMP4 X264
    frames_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv.CAP_PROP_FPS)
    # out = cv.VideoWriter('film.mp4', four_cc, fps, (frame_width, frame_height), True)
    data = []

    for i in range(frames_count):
        _, frame = capture.read()
        ad_insertion = AdInsertion(frame, logo, i, data)
        ad_insertion.build_model(config)
        ad_insertion.data_preprocessed()
    data = np.array(data)
    np.save('../data.npy', data)
    capture.release()
    # out.release()


start_time = time.time()
ad_insertion_executor('../Davy_Jones.mp4', '../orca88_logo_.png', 'configurations.yml')
print("--- %s seconds ---" % (time.time() - start_time))
