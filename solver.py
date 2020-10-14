#!/usr/bin/python

import numpy as np
from clusteringalgorithms import KMeans, Square, cubes
from sklearn.cluster import KMeans as KMeansLib
import sys
import getopt
from inspect import signature
from varname import nameof
from collections import defaultdict
import multiprocessing

available_solutions = {}


def solution(solution_name):
    def inner_decorator(func):
        available_solutions[solution_name] = func
        return func

    return inner_decorator


def begin_kmeans_thread(
        worker_num, output_dict, k, initial_points, dist_quant, central_value, join_criteria, points, verbose
):
    output = KMeans(
        k=k, initial_points=initial_points, dist_quant=dist_quant,
        central_value=central_value, join_criteria=join_criteria
    ).cluster(points, verbose=verbose)
    cur_cluster_dict = gen_cluster_dict(output, points)
    cur_cluster_dict = optimize_points(cur_cluster_dict)
    worst_cluster_distance = get_max_distance_cluster(cur_cluster_dict)[1]
    output_dict[worker_num] = tuple([worst_cluster_distance, output])


@solution("s_kmeans")
def sp_kmeans(k, points, verbose=False):
    return KMeans(
        k=k, initial_points=cubes(points, k)[0], dist_quant=1, central_value="mean", join_criteria="closest_centroid"
    ).cluster(points, verbose=verbose)


@solution("sp_kmeans")
def sp_kmeans(k, points, verbose=False):
    return_dict = {}

    i = 0
    for dist_quant in [1, 2, 3, 4, 5]:
        for central_value in ["mean", "midpoint", "static"]:
            for initial_points in cubes(points, k) + ["stacked", "furthest"]:
                for join_criteria in ["closest_centroid", "closest_furthest"]:
                    begin_kmeans_thread(
                        i, return_dict, k, initial_points, dist_quant, central_value, join_criteria, points, verbose
                    )
                    i += 1

    best = (sys.maxsize, None)
    for output in return_dict.values():
        if output[0] < best[0]:
            best = output

    return best[1]


@solution("kmeans")
def kmeans(k, points, verbose=False):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    workers = []

    i = 0
    for dist_quant in [1, 2, 3, 4, 5]:
        for central_value in ["mean", "midpoint", "static"]:
            for initial_points in cubes(points, k) + ["stacked", "furthest"]:
                for join_criteria in ["closest_centroid", "closest_furthest"]:
                    worker = multiprocessing.Process(
                        target=begin_kmeans_thread,
                        args=(
                            i, return_dict, k, initial_points, dist_quant, central_value,
                            join_criteria, points, verbose
                        )
                    )
                    workers.append(worker)
                    worker.start()
                    i += 1

    for worker in workers:
        worker.join()

    best = (sys.maxsize, None)
    for output in return_dict.values():
        if output[0] < best[0]:
            best = output
    return best[1]


@solution("kmeanslib")
def kmeanslib(k, points):
    return KMeansLib(n_clusters=k, n_init=20).fit_predict(np.asarray(points))


@solution("square")
def square(k, points):
    return Square(k=k).cluster(points)


def gen_cluster_dict(clusters, points):
    cluster_dict = dict()
    for cluster, point in zip(clusters, points):
        cluster_dict.setdefault(cluster, []).append(point)
    return cluster_dict


def organize_clusters(cluster_dict, solution):
    worst_cluster, max_distance = get_max_distance_cluster(cluster_dict)[0:2]
    cores = []
    for i, left in cluster_dict.items():
        cur_core = None
        for core in cores:
            if tuple(left) in core:
                cur_core = core
        if cur_core is None:
            cur_core = set()
            cur_core.add(tuple(left))
            cores.append(cur_core)
        for j, right in cluster_dict.items():
            combined = left.copy()
            combined.extend(right)
            if max_intracluster_distance(combined)[0] < max_distance:
                cur_core.add(tuple(right))
    free_clusters = 0
    for core in cores:
        for k in range(1, len(core)):
            points = []
            for cluster in core:
                points.extend(cluster)
            output = solution(k, points)
            new_cluster_dict = defaultdict(list)
            for cluster, point in zip(output, points):
                new_cluster_dict[cluster].append(point)
            new_max_distance = get_max_distance_cluster(new_cluster_dict)[1]
            if new_max_distance > max_distance:
                continue
            free_clusters += (len(core) - k)
            add_ind = 0
            for del_cluster in core:
                for ind, cluster in cluster_dict.items():
                    if del_cluster != tuple(cluster):
                        continue
                    if add_ind < len(new_cluster_dict):
                        cluster_dict[ind] = new_cluster_dict[add_ind]
                        add_ind += 1
                    else:
                        del cluster_dict[ind]
                    break
            break
    return free_clusters, cluster_dict


def optimize_points(cluster_dict):
    prev_p1, prev_p2 = (9999, 9999, 9999), (9999, 9999, 9999)
    while True:
        worst_cluster, cluster_distance, p1, p2 = get_max_distance_cluster(cluster_dict)
        if (p1 == prev_p1 or p1 == prev_p2) and (p2 == prev_p1 or p2 == prev_p2):
            break
        prev_p1, prev_p2 = p1, p2

        cluster_dict = move_value_to_better_cluster(worst_cluster, p1, cluster_dict)
        cluster_dict = move_value_to_better_cluster(worst_cluster, p2, cluster_dict)

    return cluster_dict


def move_value_to_better_cluster(cluster, point, cluster_dict):
    cluster.remove(point)

    best = (sys.maxsize, None)
    for key, cluster_b in cluster_dict.items():
        cluster_b.append(point)
        distance = max_intracluster_distance(cluster_b)[0]
        if best[0] > distance:
            best = (distance, key)
        cluster_b.remove(point)
    cluster_dict[best[1]].append(point)

    return cluster_dict


def read_input(file_name):
    f = open(file_name, "r")
    n = int(f.readline())
    k = int(f.readline())

    points = []
    for line in f:
        points.append(tuple([int(v) for v in line.split()]))

    f.close()

    return n, k, points


# the the Manhattan distance between two points
def get_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])


# given a cluster, find the maximum distance between any two points in the cluster
def max_intracluster_distance(cluster):
    max_distance = 0
    max_p1 = ()
    max_p2 = ()
    for p1 in cluster:
        for p2 in cluster:
            distance = get_distance(p1, p2)
            if distance > max_distance:
                max_distance = distance
                max_p1 = p1
                max_p2 = p2
    return max_distance, max_p1, max_p2


# find and return the cluster with the maximum distance between any two points
def get_max_distance_cluster(cluster_dict):
    max_cluster_distance = 0
    max_cluster = []
    bad_point_a = ()
    bad_point_b = ()
    for value in cluster_dict.values():
        cluster_distance, p1, p2 = max_intracluster_distance(value)
        if cluster_distance > max_cluster_distance:
            max_cluster_distance = cluster_distance
            max_cluster = value
            bad_point_a = p1
            bad_point_b = p2
    return max_cluster, max_cluster_distance, bad_point_a, bad_point_b


def format_output(max_cluster_distance, cluster_dict, points):
    output = [str(max_cluster_distance) + "\n"]
    for i, cluster in cluster_dict.items():
        line = ""
        for point in cluster:
            line += str(points.index(point) + 1) + " "
        output.append(line[:-1] + "\n")
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
    print(f"usage: solver.py [-h|--help] [-v|--verbose] [-m|--midpoint] --ifile ifile --ofile ofile --style [{solutions[:-1]}]")


def main(argv):
    try:
        opts, cml_args = getopt.getopt(argv, "hmvi:o:s:", ["help", "verbose", "midpoint", "ifile=", "ofile=", "style="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    ifile = ""
    ofile = ""
    styles = []
    verbose = False
    midpoint = False

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-m", "--midpoint"):
            midpoint = True
        elif opt in ("-i", "--ifile"):
            ifile = str(arg)
        elif opt in ("-o", "--ofile"):
            ofile = str(arg)
        elif opt in ("-s", "--style"):
            styles = str(arg).split(",")
        else:
            continue
        if arg in argv:
            argv.remove(arg)
        argv.remove(opt)

    if len(argv) != 0:
        usage()
        sys.exit(1)

    n, k, points = read_input(ifile)

    # DO NOT CALL ARGS DIRECTLY; FOR REFLECTION ONLY
    # Add values if you'd like functions to be able to call them
    args = {
        nameof(ifile): ifile,
        nameof(ofile): ofile,
        nameof(styles): styles,
        nameof(verbose): verbose,
        nameof(midpoint): midpoint,
        nameof(n): n,
        nameof(k): k,
        nameof(points): points
    }

    best = (sys.maxsize, None, None)
    for style in styles:
        cur_solution = available_solutions[style]
        cur_solution_sig = signature(cur_solution)
        cur_args = []
        for param in cur_solution_sig.parameters.keys():
            if param not in args:
                continue
            cur_args.append(args[param])
        cur_clusters = cur_solution(*cur_args)
        cur_clusters_dict = gen_cluster_dict(cur_clusters, points)
        cur_clusters_dict = optimize_points(cur_clusters_dict)
        worst_cluster_distance = get_max_distance_cluster(cur_clusters_dict)[1]
        if worst_cluster_distance < best[0]:
            best = tuple([worst_cluster_distance, cur_clusters_dict, cur_solution])
    cluster_dict = best[1]
    cur_solution = best[2]

    free_clusters, cluster_dict = organize_clusters(cluster_dict, cur_solution)
    if free_clusters != 0:
        print(f"WARNING:: Inefficient use of clusters. {free_clusters} free clusters unused")

    # compute the maximum distance within the clusters

    worst_cluster_distance = get_max_distance_cluster(cluster_dict)[1]

    output = format_output(worst_cluster_distance, cluster_dict, points)

    if verbose:
        print(output)

    save_output(ofile, output)


if __name__ == "__main__":
    main(sys.argv[1:])
