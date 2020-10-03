import numpy as np
from sklearn.cluster import KMeans

def read_input(file_name):
    f = open(file_name, "r")
    n = int(f.readline())
    k = int(f.readline())

    points = []
    for line in f:
        points.append([int(v) for v in line.split()])

    return n, k, points

def print_clusters(clusters, points):
    for cluster, point in zip(clusters, points):
        print(f"point: {point} cluster: {cluster}")

# the the Manhattan distance between two points
def get_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

# given a cluster, find the maximum distance between any two points in the cluster
def max_intracluster_distance(cluster):
    max_distance = 0
    for p1 in cluster:
        for p2 in cluster:
            distance = get_distance(p1, p2)
            if distance > max_distance:
                max_distance = distance
    return max_distance

# find and return the cluster with the maximum distance between any two points
def get_max_distance_cluster(cluster_dict):
    best_cluster_distance = 0
    best_cluster = []
    for value in cluster_dict.values():
        cluster_distance = max_intracluster_distance(value)
        if cluster_distance > best_cluster_distance:
            best_cluster_distance = cluster_distance
            best_cluster = value
    return best_cluster, best_cluster_distance


# read in input
n, k, points = read_input("input.txt")

# convert to numpy array
np_points = np.asarray(points)

# do KMeans clustering
clusters = KMeans(n_clusters=k, n_init=20).fit_predict(np_points)

# print out the cool clusters
print_clusters(clusters, points)

# convert output into dictionary of clusters (key = cluster_id, value = points in cluster)
cluster_dict = dict()
for cluster, point in zip(clusters, points):
    cluster_dict.setdefault(cluster, []).append(point)

# compute the maximum distance within the clusters
best_cluster, best_cluster_distance = get_max_distance_cluster(cluster_dict)

print()
print(best_cluster_distance)
print(best_cluster)
