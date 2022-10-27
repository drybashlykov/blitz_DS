import numpy as np
import cv2
from pprint import pprint
import ffmpeg
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

from precondition import cut_first_minute, play_sequence, show_frame

def threshold_filter(frame, threshold = 200, mode = 'gray'):
    if mode == 'gray':
        return np.tile(np.expand_dims(frame.mean(axis = 2), axis = 2), 3) > threshold

def filter_sequence(sequence, filter_function):
    result = np.zeros_like(sequence)
    for i in range(sequence.shape[-1]):
        result[:,:,:,i] = filter_function(sequence[:,:,:,i])
    return result

def find_player_filter(frame, threshold = 0.4, mode = 'gray'):
    if mode == 'gray':
        tmp = gaussian_filter(frame.astype(float), sigma = (5,5,0))
        return tmp > threshold

def get_player_coords(blob_frame):
    return np.array(np.where(blob_frame)).mean(axis = 1)
