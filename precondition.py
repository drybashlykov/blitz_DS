import numpy as np
import cv2
from pprint import pprint
import ffmpeg
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output

cutoff_up = 0
cutoff_down = 272
cutoff_left = 277
cutoff_right = 552
height = cutoff_down - cutoff_up
width = cutoff_right - cutoff_left

def count_frames(filename):
    counter = 0
    cap = cv2.VideoCapture(filename)
    ret = 1
    while(ret):
        ret, _ = cap.read()
        counter += 1
    counter -= 1 #the last cycle passed with no frame
    cap.release()
    return counter

def get_nbframes(filename):
    return int(ffmpeg.probe(filename)['streams'][1]['nb_frames'])

def get_frame(filename, n_frame):
    cap = cv2.VideoCapture(filename)
    for i in range(n_frame - 1):
        cap.read()
    _, frame = cap.read()
    return frame

def show_frame(frame, time = 5000):
    cv2.imshow("frame", frame)
    cv2.waitKey(time)
    cv2.destroyAllWindows()

def get_similarity(filename):
    bins = 60
    histSize = [bins]
    ranges = [0, 255]

    reference = cv2.imread("test_frame.png")[cutoff_up:cutoff_down, cutoff_left:cutoff_right]
    ref_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    ref_hue_hist = cv2.calcHist(ref_hsv, [0], None, histSize, ranges)

    hue_similarities = []
    cap = cv2.VideoCapture(filename)
    for i in range(get_nbframes(filename)):
        #print(i, end = None)
        _, frame = cap.read()
        ROI = frame[cutoff_up:cutoff_down, cutoff_left:cutoff_right]
        ROI_hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist(ROI_hsv, [0], None, histSize, ranges)
        hue_similarity = cv2.compareHist(hue_hist, ref_hue_hist, method = 0)

        hue_similarities.append(hue_similarity)
        #clear_output(wait = True)
    cap.release()
    return hue_similarities

def find_battle_start(filename, threshold = 0.2):
    bins = 60
    histSize = [bins]
    ranges = [0, 255]

    reference = cv2.imread("test_frame.png")[cutoff_up:cutoff_down, cutoff_left:cutoff_right]
    ref_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    ref_hue_hist = cv2.calcHist(ref_hsv, [0], None, histSize, ranges)

    hue_similarities = []
    cap = cv2.VideoCapture(filename)
    status = True
    similarity = 0

    counter = 0
    while(status and similarity < 0.2):
        counter += 1
        status, frame = cap.read()
        ROI = frame[cutoff_up:cutoff_down, cutoff_left:cutoff_right]
        ROI_hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist(ROI_hsv, [0], None, histSize, ranges)
        similarity = cv2.compareHist(hue_hist, ref_hue_hist, method = 0)
    cap.read()
    counter += 1
    print("Battle start frame: ", counter)
    return cap

def cut_first_minute(filename):
    cap = find_battle_start(filename)
    n_frames = 60 * 23
    sequence = np.zeros((height, width, 3, n_frames), dtype = np.uint8)
    for i in range(60 * 23):
        _, frame = cap.read()
        sequence[:,:,:,i] = frame[cutoff_up:cutoff_down, cutoff_left:cutoff_right, :]
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return sequence

def play_sequence(seq, grayscale = False):
    n_frames = seq.shape[-1]
    for i in range(n_frames):
        cv2.imshow("sequence", seq[:,:,:,i])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return 0
