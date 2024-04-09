import pandas as pd
import os
import json

"""
list the path of all the testcases
"""

base_path = 'ATBench'
data = []

cnt = 0
for root, dirs, files in os.walk(base_path):
    ancestors = root.split('/')
    if len(ancestors) != 2:
        continue
    if ancestors[-1] == 'multistep':
        continue

    for dir in dirs:
        # print(dir)

        # one testcase
        info = json.load(open(os.path.join(root, dir, 'info.json')))
        assert len(info['label']) == 1
        operator = info['label'][0]['operator']
        assert operator == ancestors[-1]

        input_path = os.path.join(root, dir, 'data.csv')
        data.append((input_path, operator))
        cnt += 1