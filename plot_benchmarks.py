import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import re
from scipy.stats import gmean
from pprint import pprint

decimal_with_parenthesis = r"([0-9\.\,]+) \([^)]*\)"
regex_dict = {
    "name": r"(\w+\[[^\]]*])",
    "min": decimal_with_parenthesis,
    "max": decimal_with_parenthesis,
    "mean": decimal_with_parenthesis,
    "stddev": decimal_with_parenthesis,
    "median": decimal_with_parenthesis,
    "iqr": None,
    "outliers": None,
    "ops": None,
    "rounds": None,
    "iterations": None,
    "spaces": r"\s+"
}


def to_float(string):
    return float(string.replace(",", ""))


def decode_name(name):
    operands = re.search(r"(binary|unary)", name).group(1)
    op = re.search(r"\[([a-z_A-Z]+)-", name).group(1)
    dtype = re.search(r"(dtype[0-9]+)", name).group(1)
    size = re.search(r"-([0-9]+)", name).group(1)
    engine = "open3d" if re.search(r"numpy", name) is None else "numpy"
    return operands, op, dtype, size, engine


if __name__ == "__main__":

    with open("benchmark_results.log", "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        spaces = r"\s+"
        line_regex = "".join([
            regex_dict["name"],
            spaces,
            regex_dict["min"],
            spaces,
            regex_dict["max"],
            spaces,
            regex_dict["mean"],
            spaces,
            regex_dict["stddev"],
            spaces,
            regex_dict["median"],
        ])
        entries = []
        for line in lines:
            match = re.search(line_regex, line)
            if match:
                entry = dict()
                entry["name"] = match.group(1).strip()
                entry["operands"], entry["op"], entry["dtype"], entry[
                    "size"], entry["engine"] = decode_name(entry["name"])
                entry["min"] = to_float(match.group(2).strip())
                entry["max"] = to_float(match.group(3).strip())
                entry["mean"] = to_float(match.group(4).strip())
                entry["stddev"] = to_float(match.group(5).strip())
                entry["median"] = to_float(match.group(6).strip())
                entries.append(entry)
        print(f"len(entries): {len(entries)}")

    # Compute geometirc mean
    binary_times = dict()
    binary_ops = [
        entry["op"] for entry in entries if entry["operands"] == "binary"
    ]
    binary_ops = sorted(list(set(binary_ops)))
    for binary_op in binary_ops:
        open3d_times = [
            entry["mean"]
            for entry in entries
            if entry["op"] == binary_op and entry["engine"] == "open3d"
        ]
        numpy_times = [
            entry["mean"]
            for entry in entries
            if entry["op"] == binary_op and entry["engine"] == "numpy"
        ]
        binary_times[binary_op] = dict()
        binary_times[binary_op]["open3d"] = gmean(open3d_times)
        binary_times[binary_op]["numpy"] = gmean(numpy_times)
    pprint(binary_times)

    unary_times = dict()
    unary_ops = [
        entry["op"] for entry in entries if entry["operands"] == "unary"
    ]
    unary_ops = sorted(list(set(unary_ops)))
    for unary_op in unary_ops:
        open3d_times = [
            entry["mean"]
            for entry in entries
            if entry["op"] == unary_op and entry["engine"] == "open3d"
        ]
        numpy_times = [
            entry["mean"]
            for entry in entries
            if entry["op"] == unary_op and entry["engine"] == "numpy"
        ]
        unary_times[unary_op] = dict()
        unary_times[unary_op]["open3d"] = gmean(open3d_times)
        unary_times[unary_op]["numpy"] = gmean(numpy_times)
    pprint(unary_times)

    # Plot
    # https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
    open3d_times = [
        binary_times[binary_op]["open3d"] for binary_op in binary_ops
    ]
    numpy_times = [binary_times[binary_op]["numpy"] for binary_op in binary_ops]

    ind = np.arange(len(binary_ops))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, open3d_times, width, color='r')
    rects2 = ax.bar(ind + width, numpy_times, width, color='y')

    ax.set_ylabel('Time (ms)')
    ax.set_title('Binary op benchmarks')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(binary_ops)

    ax.legend((rects1[0], rects2[0]), ('Open3D', 'Numpy'))

    plt.show()
