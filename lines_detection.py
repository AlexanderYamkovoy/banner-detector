import cv2 as cv
import numpy as np


def detect_lines(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    laplacian = cv.Laplacian(gray, cv.CV_64F, ksize=3)
    laplacian = np.uint8(np.absolute(laplacian))

    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    sobely = np.uint8(np.absolute(sobely))

    low_threshold = 30
    high_threshold = 60

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180
    threshold = 180  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200
    max_line_gap = 25  # maximum gap in pixels between connectible line segments
    line_image = np.copy(image) * 0

    # kernel_size = 5
    # blur_gray = cv.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    edges = cv.Canny(gray, low_threshold, high_threshold)

    lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                           min_line_length, max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            cv.line(line_image, (x1, y1), (x2, y2), color, 2)

    lines_edges = cv.addWeighted(image, 0.8, line_image, 1, 0)
    return lines_edges


def build_model(video, img_path='Set path'):
    if not video:
        img = cv.imread(img_path)
        lines = detect_lines(img)
        cv.imshow('result', lines)
        # cv.imwrite('laplacian_8u_lines.png', lines)
        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()
    else:
        capture = cv.VideoCapture('football.mp4')
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        four_cc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter('canny_theta180.avi', four_cc, 30, (frame_width, frame_height), True)
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


build_model(True, 'all_banner.png')
