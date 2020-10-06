#!/usr/bin/python

import numpy as np
from sklearn.cluster import KMeans
import sys
import getopt
from collections import defaultdict

available_solutions = {}


def solution(solution_name):
    def inner_decorator(func):
        available_solutions[solution_name] = func
        return func

    return inner_decorator


@solution("kmeans")
def kmeans(k, points):
    return KMeans(n_clusters=k, n_init=20).fit_predict(np.asarray(points))


def read_input(file_name):
    f = open(file_name, "r")
    n = int(f.readline())
    k = int(f.readline())

    points = []
    for line in f:
        points.append([int(v) for v in line.split()])

    f.close()

    return n, k, points


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
def get_max_distance_clusters(cluster_dict):
    max_cluster_distance = 0
    for value in cluster_dict.values():
        cluster_distance = max_intracluster_distance(value)
        if cluster_distance > max_cluster_distance:
            max_cluster_distance = cluster_distance
    return max_cluster_distance


def format_output(max_cluster_distance, clusters):
    clusters_point_dict = defaultdict(list)
    output = [str(max_cluster_distance) + "\n"]
    for i in range(len(clusters)):
        clusters_point_dict[clusters[i]].append(i + 1)
    for cluster, points in clusters_point_dict.items():
        line = ""
        for point in points:
            line += str(point) + " "
        line = line[:-1] + "\n"
        output.append(line)
    output[-1] = output[-1][:-1]
    return output


def save_output(ofile, output):
    with open(ofile, "w") as f:
        for line in output:
            f.write(line)


def usage():
    solutions = ""
    for i in available_solutions:
        solutions += i + "|"
    print(f"usage: solver.py [-h|--help] [-p|--print] --ifile ifile --ofile ofile --style [{solutions[:-1]}]")


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hpi:o:s:", ["help", "ifile=", "ofile=", "style="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    ifile = ""
    ofile = ""
    style = ""
    should_print = False

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif opt in ("-i", "--ifile"):
            ifile = str(arg)
        elif opt in ("-o", "--ofile"):
            ofile = str(arg)
        elif opt in ("-p", "--print"):
            should_print = True
        elif opt in ("-s", "--style"):
            style = str(arg)
        else:
            continue
        argv.remove(arg)
        argv.remove(opt)

    if len(argv) != 0:
        usage()
        sys.exit(1)

    n, k, points = read_input(ifile)
    clusters = available_solutions[style](k, points)

    # convert output into dictionary of clusters (key = cluster_id, value = points in cluster)
    cluster_dict = dict()
    for cluster, point in zip(clusters, points):
        cluster_dict.setdefault(cluster, []).append(point)

    # compute the maximum distance within the clusters
    worst_cluster_distance = get_max_distance_clusters(cluster_dict)

    output = format_output(worst_cluster_distance, clusters)

    if should_print:
        print(output)

    save_output(ofile, output)


if __name__ == "__main__":
    main(sys.argv[1:])
