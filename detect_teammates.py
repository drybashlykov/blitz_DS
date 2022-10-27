import numpy as np
import cv2
from pprint import pprint
import ffmpeg
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import imageio

from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

from precondition import cut_first_minute, play_sequence, show_frame

colors = ((0, 1, 253),
          (21, 105, 247),
          (4, 222, 238),
          (54, 214, 160),
          (54, 162, 47),
          (212, 62, 51))

def threshold_filter(frame, threshold = 200, mode = 'gray'):
    if mode == 'gray':
        return np.tile(np.expand_dims(frame.mean(axis = 2), axis = 2), 3) > threshold
    if mode == 'green':
        return np.tile(np.expand_dims(frame[:,:,0,:], axis = 2), 3) > threshold

def filter_sequence(sequence, filter_function, **kwargs):
    result = np.zeros_like(sequence)
    for i in range(sequence.shape[-1]):
        result[:,:,:,i] = filter_function(sequence[:,:,:,i], **kwargs)
    return result

def find_player_filter(frame, threshold = 0.4, s = 5, mode = 'gray'):
    if mode == 'gray':
        tmp = gaussian_filter(frame.astype(float), sigma = (s,s,0))
        return tmp > threshold

def get_player_coords(blob_frame):
    return np.array(np.where(blob_frame)).mean(axis = 1)

def band_pass_filter_frame(frame, bgr1 = (0, 150, 100), bgr2 = (255, 235, 200)): #48, 224, 148
    res_b = np.multiply(frame[:,:,0] >= bgr1[0], frame[:,:,0] <= bgr2[0])
    res_g = np.multiply(frame[:,:,1] >= bgr1[1], frame[:,:,1] <= bgr2[1])
    res_r = np.multiply(frame[:,:,2] >= bgr1[2], frame[:,:,2] <= bgr2[2])
    res_bg = np.multiply(res_b, res_g)
    return np.multiply(res_bg, res_r) #res_bgr

def band_pass_filter_seq(seq, bgr1 = (0, 150, 100), bgr2 = (255, 235, 200)): #48, 224, 148
    res_b = np.multiply(seq[:,:,0,:] >= bgr1[0], seq[:,:,0,:] <= bgr2[0])
    res_g = np.multiply(seq[:,:,1,:] >= bgr1[1], seq[:,:,1,:] <= bgr2[1])
    res_r = np.multiply(seq[:,:,2,:] >= bgr1[2], seq[:,:,2,:] <= bgr2[2])
    res_bg = np.multiply(res_b, res_g)
    return np.multiply(res_bg, res_r) #res_bgr

def color_clusters(frame, n = 6):
    coords = np.transpose(np.where(frame[:,:,0] > 200))
    k_means = KMeans(init='k-means++', n_clusters=n, n_init=10)
    k_means.fit(coords)
    k_means_labels = k_means.labels_
    res = np.zeros_like(frame)
    for i in range(6):
        for x,y in coords[k_means_labels == i]:
            res[x, y, :] = colors[i]
    return res

def initial_clusters(frame, n = 6):
    N, CA = connected_areas(frame[:,:,0])
    n_tanks = np.zeros(N, dtype = int)
    size_threshold = 200
    for i in range(N):
        upper_bound, lower_bound = np.quantile(np.where(CA == i)[0], [0.05, 0.95])
        left_bound, right_bound = np.quantile(np.where(CA == i)[1], [0.05, 0.95])
        size = (upper_bound - lower_bound) * (left_bound - right_bound)
    if size < size_threshold:
        n_tanks[i] = 1
    #print(i, size)
    centroids = []
    for i in range(N):
        if n_tanks[i] == 0:
            n_tanks[i] = int(6 - n_tanks.sum()/(n_tanks == 0).sum())
            print(n_tanks[i])
            coords = np.transpose(np.where(CA == i))
            k_means = KMeans(init='k-means++', n_clusters=n_tanks[i], n_init=10)
            k_means.fit(coords)
            # for j in range(n_tanks[i]):
            #     centroids.append(k_means.cluster_centers_[j])
            centroids += [[x, y] for x, y in k_means.cluster_centers_]
        else:
            centroids.append(np.array(np.where(CA == i)).mean(axis = 1))
    centroids = np.array(centroids)
    return centroids

def find_clusters(frame, init_centers, n = 6):
    coords = np.transpose(np.where(frame[:,:,0] > 200))
    k_means = KMeans(init=init_centers, n_clusters=n, n_init = 1)
    k_means.fit(coords)
    k_means_labels = k_means.labels_
    res = np.zeros_like(frame)
    for i in range(6):
        for x,y in coords[k_means_labels == i]:
            res[x, y, :] = colors[i]
    return k_means.cluster_centers_, res

def clusters_sequence(seq, n = 6):
    res_seq = np.zeros_like(seq)
    centers = initial_clusters(seq[:,:,:,10])
    for i in range(seq.shape[-1]):
        centers, res_frame = find_clusters(seq[:,:,:,i], centers)
        res_seq[:,:,:,i] = res_frame
    return res_seq

def is_in(array, coords):
    for i in range(array.shape[0]):
        if (coords == array[i]).all():
            return True
    return False

def connected_areas(T):
    x_max = T.shape[0]
    y_max = T.shape[1]
    all_points = np.transpose(np.where(T))
    area_map = -1 * np.ones_like(T)
    counter = 0
    counter_complexity = 0
    for x, y in all_points:
        counter_complexity += 1
        if area_map[x, y] == -1:
            q = []
            visited = np.zeros((x_max, y_max)).astype(bool)
            visited[x, y] = True
            q.append((x, y))
            while q:
                cur_x, cur_y = q.pop()
                for dx, dy in [(+1, 0), (-1, 0), (0, +1), (0, -1)]:
                    next_x = cur_x + dx
                    next_y = cur_y + dy
                    if next_x >= 0 and next_y >= 0 and next_x < x_max and next_y < y_max\
                        and is_in(all_points, [next_x, next_y]) and visited[next_x, next_y] == False:
                        visited[next_x, next_y] = True
                        q.append((next_x, next_y))

            area_map[visited] = counter
            #print("cluster ", counter)
            counter += 1
    #print("counter_complexity: ", counter_complexity)
    return counter, area_map
