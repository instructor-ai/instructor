import pandas as pd


def flatten_dict(d, parent_key="", sep="_"):
    """
    Flatten a nested dictionary.

    :param d: The nested dictionary to flatten.
    :param parent_key: The base key to use for the flattened keys.
    :param sep: Separator to use between keys.
    :return: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dicts_to_df(list_of_dicts):
    """
    Convert a list of dictionaries to a pandas DataFrame.

    :param list_of_dicts: List of dictionaries, potentially nested.
    :return: A pandas DataFrame representing the flattened data.
    """
    # Flatten each dictionary and create a DataFrame
    flattened_data = [flatten_dict(d) for d in list_of_dicts]
    return pd.DataFrame(flattened_data)
