#!/usr/bin/python

# I wrote this specifically for my computer. It may not work for yours. Please don't change it though
# I wrote this specifically for my computer. It may not work for yours. Please don't change it though
# I wrote this specifically for my computer. It may not work for yours. Please don't change it though
# I wrote this specifically for my computer. It may not work for yours. Please don't change it though
# I wrote this specifically for my computer. It may not work for yours. Please don't change it though
# I wrote this specifically for my computer. It may not work for yours. Please don't change it though
import os
import sys
import getopt


def usage():
    print("./run_all [-h:--help] [-p|--path <algobowl folder>]")


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "p:h", ["path=", "help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    path = "../../algobowl/"
    input_format = "input_group%d.txt"
    output_format = "./output/output_group%d.txt"

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif opt in ("-p", "--path"):
            path = str(arg)
        else:
            continue

    for num_group in [196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 241]:
        input = path + (input_format % num_group)
        output = path + (output_format % num_group)
        command = f'./solver.py --ifile {input} --ofile {output} --style kmeans'
        os.system(command)

if __name__ == "__main__":
    main(sys.argv[1:])