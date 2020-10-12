#!/usr/bin/python

import sys
import getopt
import random
from math import pi, cos, sin
from numpy import arange

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

@input_style("randomizerFunction") 
def randomizerFunction(n, k): 
    output = []
    for i in range(n):
        i = random.randrange(-1000, 1000, 1) 
        j = random.randrange(-1000, 1000, 1)
        k = random.randrange(-1000, 1000, 1)
        output.append([i, j, k])
    return output


@input_style("smd")
def super_megadeath(n, k):
    planets = [
        [0, 0, 0, 5],
        [0, 0, 0, 15],
        [0, 0, 0, 25],
        [0, 200, 0, 10],
        [-10, 200, 0, 3],
        [10, 200, 200, 3],
        [0, -300, 0, 10],
        [0, -600, 0, 4],
        [85, 85, 0, 70, 0.35],
        [120, 120, 0, 40],
        [135, 135, 0, 10],
        [135, 135, 100, 20],
        [135, 135, 180, 50, 0.4],
        [135, 200, 180, 30],
        [135, 230, 180, 10],
        [900, 900, 900, 6]
    ]
    output = set()
    output.add((-1000, 1000, 1000))
    output.add((1000, -1000, -1000))
    output.add((-1000, -1000, -1000))
    output.add((1000, 1000, -1000))
    for planet in planets:
        output.update(sphere(*planet))
    return list(output)


@input_style("armyof5")
def armyof5(n, k):
    output = set()
    output.update(sphere(0, 0, 0, 25))

    output.update(sphere(1000, -1000, 1000, 2))
    return list(output)


@input_style("square")
def square(n, k):
    output = []
    for y in range(0, 50, 10):
        for x in range(0, 50, 10):
            output.append((x, y, 0))
    return output


@input_style("sphere")
def sphere(n, k):
    output = set()
    output.update(sphere(0, 0, 0, 100, 0.1))
    return list(output)


def sphere(x1=0, y1=0, z1=0, r=0, density=0.6):
    output = []
    for theta in arange(0, 2*pi, density):
        for phi in arange(0, pi, density):
            x = int(cos(theta) * sin(phi) * r + x1)
            y = int(sin(theta) * sin(phi) * r + y1)
            z = int(cos(phi) * r + z1)
            output.append(tuple([x, y, z]))
    return output


def randomize(output):
    random.shuffle(output)


def usage():
    inputs = ""
    for i in available_inputs:
        inputs += i + "|"
    print(f"usage: generate-input.py [-r|--randomize] --ofile ofile --lines num_lines --clusters num_clusters --style [{inputs[:-1]}]")
    print("2 <= num_clusters <= 20")
    print("3 <= num_lines <= 1000")
    print("num_clusters < num_lines")


def main(argv):
    random.seed() 
    try:
        opts, args = getopt.getopt(argv, "n:k:s:rho:", ["randomize", "lines=", "clusters=", "style=", "help", "ofile="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    n = -1
    k = -1
    ofile = ""
    style = ""
    is_randomized = False

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
        elif opt in ("-r", "--randomize"):
            is_randomized = True
        else:
            continue

    if 20 < k < 2 or k >= n or 1000 < n < 3 or style not in available_inputs or ofile == "":
        usage()
        sys.exit(1)

    output = available_inputs[style](n, k)
    if is_randomized:
        randomize(output)

    with open(ofile, "w+") as file:
        raw = ""
        file.write(str(len(output)) + "\n")
        file.write(str(k) + "\n")
        for point in output:
            for cord in point:
                raw += str(cord) + " "
            raw = raw[:-1] + "\n"
        file.write(raw[:-1])


if __name__ == "__main__":
    main(sys.argv[1:])
