import csv
import os
import math


def is_file_empty(file_name):
    """ Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return os.path.isfile(file_name) == 0


for i in range(1, 100):
    j = 5*math.sqrt(i*2)
    csv_file = 'example.csv'
    fieldnames = ['x', 'y']
    row = {fieldnames[0]: i, fieldnames[1]: j}
    firstCreate = False

    if is_file_empty(csv_file):
        firstCreate = True

    with open(csv_file, 'a+') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        if firstCreate:
            csv_writer.writeheader()
        csv_writer.writerow(row)
