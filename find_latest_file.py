import os
import glob
import pandas as pd
import datetime


def find_latest_file(directory, extension, key_words=None):
    path_to_cwd = os.getcwd()
    csvs_in_directory = glob.glob(os.path.join(path_to_cwd + '\\' + directory.replace('/', '\\'), '*' + extension))
    csvs_with_keywords = [el for el in csvs_in_directory if key_words in el] if key_words is not None else csvs_in_directory   # to reject "patterns.pkl"
    if len(csvs_with_keywords) == 0:
        print(f'NO FILES FOUND IN DIRECTORY: "{directory}" with EXTENSION "{extension}"')
        return None, None
    csv_dict = {}
    for csv in csvs_with_keywords:
        if (key_words is not None) & (key_words not in csv):
            continue
        if key_words:
            date = pd.to_datetime(csv.split('\\')[-1].split(extension)[0].split(key_words)[1]).date()
        else:
            date = pd.to_datetime(csv.split('\\')[-1].split(extension)[0]).date()
        csv_dict[date] = csv
    return csv_dict[max(csv_dict.keys())], max(csv_dict.keys())


def find_previous_file(directory, extension, key_words=None):
    path_to_cwd = os.getcwd()
    csvs_in_directory = glob.glob(os.path.join(path_to_cwd + '\\' + directory.replace('/', '\\'), '*' + extension))
    csvs_with_keywords = [el for el in csvs_in_directory if key_words in el] if key_words is not None else csvs_in_directory   # to reject "patterns.pkl"
    if len(csvs_with_keywords) == 0:
        print(f'NO FILES FOUND IN DIRECTORY: "{directory}" with EXTENSION "{extension}"')
        return None, None
    csv_dict = {}
    for csv in csvs_with_keywords:
        if (key_words is not None) & (key_words not in csv):
            continue
        if key_words:
            date = pd.to_datetime(csv.split('\\')[-1].split(extension)[0].split(key_words)[1]).date()
        else:
            date = pd.to_datetime(csv.split('\\')[-1].split(extension)[0]).date()
        csv_dict[date] = csv
    list_of_keys = list(csv_dict.keys())
    list_of_keys.remove(datetime.datetime.now().date())
    if len(list_of_keys) != 0:
        return csv_dict[max(list_of_keys)], max(list_of_keys)
    else:
        return None, None
