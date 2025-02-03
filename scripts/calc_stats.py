# This file calculates mean and std from csv files typically from TensorBoard log files

# MIT License
# 
# Copyright (c) 2023 Rafael Redondo, Eurecat.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import csv
import numpy as np

log_freq = 200
last_iters = 20e3
last_log_entries = int(last_iters/log_freq)
path = './log_files'
log_files = os.listdir(path)

means = []
stds = []
maxs = []
mins = []
for file in log_files:
    print(f'Processing {file}')
    with open(os.path.join(path, file), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)   # skip first row
        values = []
        for row in reader:
            values.append(float(row[-1]))
    
    last_values = values[last_log_entries:]
    mean = np.mean(last_values)
    std  = np.std(last_values)
    max  = np.max(last_values)
    min  = np.min(last_values)

    means.append(mean)
    stds.append(std)
    maxs.append(max)
    mins.append(min)


with open('log_files-stats.csv', 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['filename','mean','std','max','min'])
    for file, mean, std, max, min in zip(log_files, means, stds, maxs, mins):
        writer.writerow([file, f'{mean:.2f}', f'{std:.2f}', f'{max:.2f}', f'{min:.2f}'])

