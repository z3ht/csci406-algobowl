#!/usr/bin/python

import sys
import getopt
import random

available_inputs = {}


def input_style(style_type):
    def inner_decorator(func):
        available_inputs[style_type] = func
        return func
    return inner_decorator


@input_style("unequal")
def unequal(n, k):
    output = [[0, 0, 0]]
    cluster_delta = 3
    output.extend([[point[0] + cluster_delta, point[1] + cluster_delta, point[2] + cluster_delta] for point in clusters(n - 1, k-1)])
    return output


@input_style("uniform")
def uniform(n, k):
    output = []
    for i in range(n):
        output.append([i, i, i])
    return output


@input_style("clusters")
def clusters(n, k):
    cluster_delta = 100
    cur_cluster_delta = -1 * cluster_delta
    output = []
    for i in range(n):
        if i % (n // k) == 0:
            cur_cluster_delta += cluster_delta
        output.append([cur_cluster_delta + i, cur_cluster_delta + i, cur_cluster_delta + i])
    return output


@input_style("onlyx_uniform")
def onlyx_uniform(n, k):
    output = []
    for i in range(n):
        output.append([i, 0, 0])
    return output


def randomize(list_of_points):
    for i in range(7):
        random.shuffle(list_of_points)
    return list_of_points

def usage():
    inputs = ""
    for i in available_inputs:
        inputs += i + "|"
    print(f"usage: generate-input.py --ofile ofile --lines num_lines --clusters num_clusters --style [{inputs[:-1]}]")
    print("2 <= num_clusters <= 20")
    print("3 <= num_lines <= 1000")
    print("num_clusters < num_lines")


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:k:s:ho:", ["lines=", "clusters=", "style=", "help", "ofile="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    n = -1
    k = -1
    ofile = ""
    style = ""

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif opt in ("-n", "--lines"):
            n = int(arg)
        elif opt in ("-k", "--clusters"):
            k = int(arg)
        elif opt in ("-s", "--style"):
            style = str(arg)
        elif opt in ("-o", "--ofile"):
            ofile = str(arg)
        else:
            continue

    if 20 < k < 2 or k >= n or 1000 < n < 3 or style not in available_inputs or ofile == "":
        usage()
        sys.exit(1)

    output = available_inputs[style](n, k)

    with open(ofile, "w+") as file:
        raw = ""
        file.write(str(n) + "\n")
        file.write(str(k) + "\n")
        for point in output:
            for cord in point:
                raw += str(cord) + " "
            raw = raw[:-1] + "\n"
        file.write(raw[:-1])


if __name__ == "__main__":
    main(sys.argv[1:])
