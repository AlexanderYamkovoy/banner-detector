from ad_insertion import AdInsertion
import cv2 as cv


def ad_insertion_executor():
    capture = cv.VideoCapture(video_path)
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    four_cc = cv.VideoWriter_fourcc(*'FMP4')  # XVID FMP4 X264
    frames_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv.CAP_PROP_FPS)
    out = cv.VideoWriter('draw_quadrilaterals_fragments.avi',
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
        frames_ranges = data_cleaning('data.csv', 60, f)

    for i in range(frames_count):
        _, frame = capture.read()
        if i in frames_ranges:
            print('Processing frame {}'.format(i))
            drawing_contours('data.csv', i, frame)
            out.write(frame)

    capture.release()
    out.release()


start_time = time.time()
execution(local_video_path, False)
print("--- %s seconds ---" % (time.time() - start_time))
