import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import os
import re
from scipy.stats import gmean

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


def simplify_name(name):
    return name.replace("test_", "").replace("_ops", "").replace("_ew", "")


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
                entry["name"] = simplify_name(match.group(1).strip())
                entry["min"] = match.group(2).strip()
                entry["max"] = match.group(3).strip()
                entry["mean"] = match.group(4).strip()
                entry["stddev"] = match.group(5).strip()
                entry["median"] = match.group(6).strip()
                entries.append(entry)
        print(f"len(entries): {len(entries)}")

    # Plotting
    print(entries[0])
