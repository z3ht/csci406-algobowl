import numpy as np


class KMeans:

    def __init__(self, k):
        self.k = k

    points = []
    centroids = dict()
    k = 1
    max_iterations = 100

    def cluster(self, points, verbose=False):
        self.points = points

        self.initial_points(points)

        if verbose:
            print(f"Here are the initial centroids: {self.centroids.keys()}")

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
                c = self.find_closest_centroid(p)
                # assign the point to that cluster
                self.centroids[c].add(p)

            for c in list(self.centroids.keys()):
                cluster = self.centroids.get(c)
                cluster_mean = self.get_mean_point(cluster)
                self.centroids[cluster_mean] = self.centroids.pop(c)
            i += 1

        cluster_idx = 0
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

    def initial_points(self, points):
        # The odds of this being an actual centroid are monumentally low. It is very important it is not
        self.centroids[tuple([-1871237723123, -1871237723123, -1871237723123])] = set(
            tuple([-1871237723123, -1871237723123, -1871237723123])
        )

        while len(self.centroids) <= self.k:
            max_dist = -1
            next_furthest = None
            for point in points:
                centroid = self.find_closest_centroid(point)
                dist = self.get_distance(point, centroid)
                if dist > max_dist:
                    max_dist = dist
                    next_furthest = point
            self.centroids[next_furthest] = set(next_furthest)

        self.centroids.pop(tuple([-1871237723123, -1871237723123, -1871237723123]))

    # the the Manhattan distance between two points
    def get_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

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
        return (x_count / len(cluster), y_count / len(cluster), z_count / len(cluster))

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

    # given a cluster, find the maximum distance between any two points in the cluster
    def max_intracluster_distance(self, cluster):
        max_distance = 0
        for p1 in cluster:
            for p2 in cluster:
                distance = self.get_distance(p1, p2)
                if distance > max_distance:
                    max_distance = distance
        return max_distance