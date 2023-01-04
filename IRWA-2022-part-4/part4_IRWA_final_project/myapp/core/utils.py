import json


def load_json_file(path):
    """Load JSON content from file in 'path'

    Parameters:
    path (string): the file path

    Returns:
    JSON: a JSON object
    """

    # Load the file into a unique string
    with open(path) as fp:
        text_data = fp.readlines()[0]
    # Parse the string into a JSON object
    json_data = json.loads(text_data)
    return json_data
