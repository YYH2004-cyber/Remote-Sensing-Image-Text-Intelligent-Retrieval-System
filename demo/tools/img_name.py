import json
from pprint import pprint

with open('/home/dhm/python_project/demo/tools/dataset_rsicd.json', 'r') as f:
    data = json.load(f)
    
    pprint(data['images'][0].keys())
    pprint(data['images'][0])

    label = dict