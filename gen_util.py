import json

def read_app_config(filename):
    with open(filename) as json_file:
        config = json.load(json_file)
    return config