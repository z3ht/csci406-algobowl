import collections
import sys
import random
from math import floor, ceil

import numpy as np
import multiprocessing


def get_min_maxs(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]

    return min(x), max(x), min(y), max(y), min(z), max(z)


def base(max, min, total):
    return (max - min)/total


def cubes(points, k):
    x_min, x_max, y_min, y_max, z_min, z_max = get_min_maxs(points)

    best = sys.maxsize
    outputs = set()
    for x in range(k + 1):
        for y in range(k + 1):
            for z in range(k + 1):
                square_size = x**2 + y**2 + z**2
                if x * y * z == k and square_size <= best:
                    if square_size < best:
                        outputs.clear()
                        best = square_size
                    outputs.add(tuple([x, y, z]))
                    outputs.add(tuple([x, z, y]))
                    outputs.add(tuple([y, x, z]))
                    outputs.add(tuple([y, z, x]))
                    outputs.add(tuple([z, y, x]))
                    outputs.add(tuple([z, x, y]))

    results = []
    for best in outputs:
        result = {}
        for x in range(best[0]):
            for y in range(best[1]):
                for z in range(best[2]):
                    val = tuple([int(base(x_max, x_min, best[0]) * ((x + 0.5) - best[0] / 2)),
                                int(base(y_max, y_min, best[1]) * ((y + 0.5) - best[1] / 2)),
                                int(base(z_max, z_min, best[2]) * ((z + 0.5) - best[2] / 2))])
                    value = set()
                    value.add(val)
                    result[val] = value
        results.append(result)
    return results


class KMeans:

    def __init__(self, k, initial_points, dist_quant, central_value, join_criteria):
        """
        Initialize KMeans estimator

        Parameters
        ----------
        :param k: Number of clusters
        :param dist_quant: Distance metric (1=manhattan, 2=euclidean, ...)
        :param central_value: Available options: 'midpoint', 'mean'
        :param initial_points: Initial points creation function. Options: 'furthest', 'stacked'
        :param join_criteria: Join points based off: 'closest_centroid', 'closest_furthest'
        """
        self.k = k
        self.dist_quant = dist_quant

        linkage_criteria_dict = {
            "midpoint": self.get_center_point,
            "mean": self.get_mean_point,
            "static": self.keep_centroid
        }
        if isinstance(central_value, str) and central_value.lower() in linkage_criteria_dict:
            self.central_value = linkage_criteria_dict[central_value.lower()]
        else:
            self.central_value = central_value

        join_criteria_dict = {
            "closest_centroid": self.find_closest_centroid,
            "closest_furthest": self.find_min_furthest
        }
        if isinstance(join_criteria, str) and join_criteria.lower() in join_criteria_dict:
            self.join_criteria = join_criteria_dict[join_criteria.lower()]
        else:
            self.join_criteria = join_criteria

        initial_points_dict = {
            "furthest": self.furthest_initial_points,
            "stacked": self.stacked_initial_points,
            "centered": self.centered_initial_points,
            "random": self.random_initial_points
        }
        if isinstance(initial_points, str) and initial_points.lower() in initial_points_dict:
            self.initial_points = initial_points_dict[initial_points.lower()]
        else:
            self.initial_points = initial_points

    centroids = dict()
    k = 1
    max_iterations = 300

    def cluster(self, points, verbose=False):
        self.points = points

        self.centroids = {}
        if isinstance(self.initial_points, collections.Callable):
            self.initial_points(points)
        else:
            self.centroids = self.initial_points

        if verbose:
            print(f"Here are the initial centroids: {self.centroids.keys()}")

        previous_centroids = []
        # Repeat steps 4 and 5 until convergence or until the end of a fixed number of iterations
        i = 0
        while i < self.max_iterations:
            # if verbose output is requested print out all intra-cluster distances on this iteration
            if verbose and i > 0:
                distance_string = ""
                for cluster, j in zip(self.centroids.values(), range(len(self.centroids))):
                    distance_string += f" cluster{j + 1} distance = {self.max_intracluster_distance(cluster)}"
                print(f"iteration {i}: {distance_string}")

            # clear dictionary values
            for c in list(self.centroids.keys()):
                self.centroids[c] = set()

            for p in self.points:
                # find the nearest centroid(c_1, c_2 .. c_k)
                c = self.join_criteria(p)
                # assign the point to that cluster
                self.centroids[c].add(p)

            for c in list(self.centroids.keys()):
                cluster = self.centroids.get(c)
                cluster_mean = self.central_value(cluster)
                if cluster_mean != -1:
                    self.centroids[cluster_mean] = self.centroids.pop(c)
            i += 1

            # if the centroids do not change they never will, break from the loop
            if list(self.centroids.keys()) == previous_centroids:
                break
            previous_centroids = list(self.centroids.keys())

        result_list = []

        for point in points:
            cluster_idx = 0
            for cluster in self.centroids.values():
                if point in cluster:
                    result_list.append(cluster_idx)
                cluster_idx += 1

        assert len(result_list) == len(points)

        if verbose:
            print(f"finished in {i} iterations")

        return result_list

    # the the Manhattan distance between two points
    def get_distance(self, p1, p2):
        return (abs(p1[0] - p2[0]) ** self.dist_quant + abs(p1[1] - p2[1]) ** self.dist_quant + abs(p1[2] - p2[2]) ** self.dist_quant ) ** float(1/self.dist_quant)

    def get_mean_point(self, cluster):
        if len(cluster) == 0:
            return -1
        x_count = 0
        y_count = 0
        z_count = 0
        for point in cluster:
            x_count += point[0]
            y_count += point[1]
            z_count += point[2]
        return x_count / len(cluster), y_count / len(cluster), z_count / len(cluster)

    def get_center_point(self, cluster):
        if len(cluster) == 0:
            return -1
        min_point = min(cluster)
        max_point = max(cluster)
        return ((min_point[0] + max_point[0]) / 2), ((min_point[1] + max_point[1]) / 2), (
                    (min_point[2] + max_point[2]) / 2)

    def keep_centroid(self, cluster):
        return -1

    # find the closest centroid to a given point
    def find_closest_centroid(self, point):
        min_distance = np.inf
        closest_centroid = ()
        for c in self.centroids.keys():
            distance = self.get_distance(c, point)
            if distance <= min_distance:
                min_distance = distance
                closest_centroid = c
        return closest_centroid

    def find_min_furthest(self, point):
        min_furthest = (sys.maxsize, None)
        for c, c_points in self.centroids.items():
            max_furthest = self.get_distance(c, point)
            for far_point in c_points:
                cur_dist = self.get_distance(far_point, point)
                if cur_dist > max_furthest:
                    max_furthest = cur_dist
            if max_furthest < min_furthest[0]:
                min_furthest = tuple([max_furthest, c])
        return min_furthest[1]

    # given a cluster, find the maximum distance between any two points in the cluster
    def max_intracluster_distance(self, cluster):
        max_distance = 0
        for p1 in cluster:
            for p2 in cluster:
                distance = self.get_distance(p1, p2)
                if distance > max_distance:
                    max_distance = distance
        return max_distance

    def stacked_initial_points(self, points):
        x_min, x_max, y_min, y_max, z_min, z_max = get_min_maxs(points)

        base = (z_max - z_min)/self.k + z_min

        z_points = [base * i for i in range(self.k)]

        for z in z_points:
            self.centroids[tuple([(x_max - x_min)/2, (y_max - y_min)/2, z])] = set(tuple([(x_max - x_min)/2, (y_max - y_min)/2, z]))

    def centered_initial_points(self, points):
        cs = [tuple([-self.k//2, 0, i]) for i in range(self.k)]
        for c in cs:
            self.centroids[c] = set()

    def random_initial_points(self, points):
        x_min, x_max, y_min, y_max, z_min, z_max = get_min_maxs(points)
        for _ in range(self.k):
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            z = random.uniform(z_min, z_max)
            c = tuple([x, y, z])
            self.centroids[c] = set()

    def furthest_initial_points(self, points):
        # The odds of this being an actual centroid are monumentally low. It is very important it is not
        self.centroids = {
            tuple([-1871237723123, -1871237723123, -1871237723123]): set(tuple([-1871237723123, -1871237723123, -1871237723123]))
        }

        while len(self.centroids) <= self.k:
            max_dist = -1
            next_furthest = None
            for point in points:
                centroid = self.find_closest_centroid(point)
                dist = self.get_distance(point, centroid)
                if dist > max_dist:
                    max_dist = dist
                    next_furthest = point
            if next_furthest is not None:
                self.centroids[next_furthest] = set(next_furthest)

        self.centroids.pop(tuple([-1871237723123, -1871237723123, -1871237723123]))
