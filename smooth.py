from scipy.signal import savgol_filter
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np


def smooth_series(min_window, max_window, poly_degree, threshold, series_):
    
    best_diff = 0
    best_series = []
    wnd = 0
    
    for wnd_size in range(min_window, max_window):
        if wnd_size % 2 == 0:
            continue
        new_series = savgol_filter(series_, wnd_size, poly_degree)
        if max(abs(new_series-series_)) < threshold:
            best_diff = max(abs(new_series-series_))
            best_series = new_series
            wnd = wnd_size
                
    return best_series, best_diff, wnd


data = pd.read_csv('')

columns = ['frame', 'TL_x', 'TL_y', 'BL_x', 'BL_y', 'BR_x', 'BR_y', 'TR_x', 'TR_y']
smoothed_data = pd.DataFrame(columns=columns)
smoothed_data['frame'] = data['frame'].astype(int)

for column in columns[1:]:
    series, _, _ = smooth_series(5, 35, 4, 6.5, data[column])
    smoothed_data[column] = series


smoothed_data.to_csv('', index=False)
plt.plot(data[''])
plt.plot(smoothed_data[''])
plt.title('Corner coordinates before/after smoothing')
plt.show()
