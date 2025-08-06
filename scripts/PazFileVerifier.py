import pandas as pd
import re
import os

"""
Python script that extracts the list of PAZ files from stationlist.data file and then check if they exist!
"""


def extract_paz_files(filepath):
    paz_files = set()

    with open(filepath, 'r') as file:
        lines = file.readlines()

        for line in lines[1:]:  # Skip header
            columns = line.strip().split('\t')
            if len(columns) >= 14:
                pazfile = columns[13]
                if pazfile.endswith('.paz'):
                    paz_files.add(pazfile)

    return sorted(paz_files)


def check_paz_files_exist(paz_files, directory):
    print(f"Checking existence of .paz files in: {directory}\n")

    for paz in paz_files:
        full_path = os.path.join(directory, paz)
        if os.path.isfile(full_path):
            print(f"[FOUND]     {paz}")
        else:
            print(f"[MISSING]   {paz}")


if __name__ == "__main__":
    # Adjust these paths as needed
    stationlist_path = "../inputs/stationlist.dat"
    paz_dir = "../inputs/paz/"

    paz_files = extract_paz_files(stationlist_path)
    check_paz_files_exist(paz_files, paz_dir)
