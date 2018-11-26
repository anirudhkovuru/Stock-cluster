import numpy as np
import random
import glob
import pandas as pd
import multiprocessing as mp
import pickle


def euclidean_dist(t1, t2):
    dist = 0
    for j in range(len(t1)):
        dist = dist + (t1[j] - t2[j]) ** 2
    return dist


def lb_keogh(s1, s2, r):
    lb_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = np.amin(s2[(ind - r if ind - r >= 0 else 0):(ind + r)], axis=0)
        upper_bound = np.amax(s2[(ind - r if ind - r >= 0 else 0):(ind + r)], axis=0)

        for j in range(len(i)):
            if i[j] > upper_bound[j]:
                lb_sum = lb_sum + (i[j] - upper_bound[j]) ** 2
            elif i[j] < lower_bound[j]:
                lb_sum = lb_sum + (i[j] - lower_bound[j]) ** 2

    return np.sqrt(lb_sum)


def dtw_distance(s1, s2, w=None):
    """
    Calculates dynamic time warping Euclidean distance between two
    sequences. Option to enforce locality constraint for window w.
    """
    dtw = {}
    if w:
        w = max(w, abs(len(s1) - len(s2)))

        for i in range(-1, len(s1)):
            for j in range(-1, len(s2)):
                dtw[(i, j)] = float('inf')

    else:
        for i in range(len(s1)):
            dtw[(i, -1)] = float('inf')
        for i in range(len(s2)):
            dtw[(-1, i)] = float('inf')

    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        if w:
            for j in range(max(0, i - w), min(len(s2), i + w)):
                dist = euclidean_dist(s1[i], s2[j])
                dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])
        else:
            for j in range(len(s2)):
                dist = euclidean_dist(s1[i], s2[j])
                dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])

    return np.sqrt(dtw[len(s1) - 1, len(s2) - 1])


class Tscluster:
    def __init__(self, num_cluster):
        """
        num_cluster is the number of clusters for the k-means algorithm
        assignments holds the assignments of data points (indices) to clusters
        centroids holds the centroids of the clusters
        """
        self.num_cluster = num_cluster
        self.assignments = {}
        self.centroids = [0] * num_cluster

    def k_means_cluster(self, data, num_iter, w, callback=None, progress=False):
        """
        k-means clustering algorithm for time series data.  dynamic time warping Euclidean distance
        used as default similarity measure.
        """
        if callback is not None:
            with open(callback, "rb") as ckpt:
                temp = pickle.load(ckpt)
                for k, v in temp.items():
                    self.centroids[k] = v
        else:
            self.centroids = random.sample(list(data.values()), self.num_cluster)

        for n in range(num_iter):
            if progress:
                print('Iteration : ' + str(n+1))
            db = {}
            file_inp = open('chckpt_' + str(n+1),'wb')
            for k, c in enumerate(self.centroids):
                print("cluster " + str(k) + ":")
                print(c)
                file_inp = open('ckpt_test', 'wb')
                db[c] = j 
            pickle.dump(db, file_inp)

            # assign data points to clusters
            self.assignments = {}

            args = [(t, i, w) for t, i in data.items()]
            pool = mp.Pool(processes=mp.cpu_count())
            assignments = pool.map(self.k_means_util_multiprocessing, args)
            pool.close()
            pool.join()

            for assignment in assignments:
                if assignment[0] not in self.assignments:
                    self.assignments[assignment[0]] = []
                self.assignments[assignment[0]].append(assignment[1])

            # recalculate the centroids of clusters
            temp_centroids = self.centroids
            for key in self.assignments:
                print(key)
                if key != "Outlier":
                    cluster_sum = np.zeros(data[self.assignments[key][0]].shape)
                    for k in self.assignments[key]:
                        cluster_sum = cluster_sum + data[k]
                    self.centroids[key] = cluster_sum / len(self.assignments[key])

            if temp_centroids == self.centroids:
                print("Convergence reached!")
                break

    def k_means_util_multiprocessing(self, args):
        assign = []
        min_dist = float('inf')
        closest_cluster = "Outlier"
        for c_ind, j in enumerate(self.centroids):
            if lb_keogh(args[1], j, 5) < min_dist:
                cur_dist = dtw_distance(args[1], j, args[2])
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_cluster = c_ind

        assign.append(closest_cluster)
        assign.append(args[0])
        return assign


if __name__ == '__main__':
    mp.freeze_support()

    files = glob.glob('./preprocessed_stocks/*')
    stock_data = {}
    count = 0

    for i, f in enumerate(files):
        # read data for each stock
        t_data = pd.read_csv(f)

        # extract data from columns
        oc = np.reshape(np.asarray(t_data['o/c'].tolist()), (-1, 1))
        volume = np.reshape(np.asarray(t_data['volume'].tolist()), (-1, 1))
        high = np.reshape(np.asarray(t_data['high'].tolist()), (-1, 1))
        low = np.reshape(np.asarray(t_data['low'].tolist()), (-1, 1))

        # get stock ticker name and create 2D numpy array for values
        ticker = t_data['name'].tolist()[0]
        ts_data = np.hstack((oc, volume, high, low))

        # handle missing data by ignoring
        if ts_data.shape[0] == 1259:
            stock_data[ticker] = ts_data

    cluster = Tscluster(8)
    cluster.k_means_cluster(stock_data, 1, 2, callback='chckpt_20', progress=True)
