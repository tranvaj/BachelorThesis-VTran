import os
#from pathlib import Path
import json
import configparser

config = configparser.ConfigParser()
config.read('config.cfg')

TEXT_PREFIX = config['POC_Dataset']['TEXT_PREFIX']
FOLDER_SUFFIX = config['POC_Dataset']['FOLDER_SUFFIX']
SUMMARY_PREFIX = config['POC_Dataset']['SUMMARY_PREFIX']

PROMPT = config['POC_Dataset']['PROMPT']
SUMMARY_TOTAL = config['POC_Dataset']['SUMMARY_TOTAL']
SUMMARY_TOTAL_PROMPT = config['POC_Dataset']['SUMMARY_TOTAL_PROMPT']
DATASET_POC = config['POC_Dataset']['DATASET_POC']
DATASET_POC_P = config['POC_Dataset']['DATASET_POC_P']
TEXT_LOC = config['POC_Dataset']['TEXT_LOC']

def process_folder(folder_path, text_prefix, summary_prefix, max_dirs=-1):
    """
    Process a folder containing pages (text files) 
    and generate an issue dictionary containing the pages and the summary total.

    Args:
        folder_path (str): The path to the folder containing the pages (text files).
        text_prefix (str): The prefix used to identify the text files.
        summary_prefix (str): The prefix used to identify the summary files.
        max_dirs (int, optional): The maximum number of directories to process. Defaults to -1, which means all directories will be processed.

    Returns:
        tuple: A tuple containing two elements:
            - data (dict): The issue as dictionary containing the pages and the summary total.
            - data2 (list): A list containing just the pages.

    """
    dirs = os.listdir(folder_path)
    data_poc = {}
    data_poc_p = []
    dir_count = 0
    for dir in dirs:
        if os.path.isdir(folder_path):
            dir_count += 1
            if max_dirs != -1 and dir_count > max_dirs:
                break
            files = os.listdir(folder_path + "/" + dir)
            print(dir)
            dataset_pair_list = []
            page_num = 1
            files =  sorted(files)

            summary_total = ""
            for file in files:
                if not file.endswith('.txt'):
                    continue
                if not file.startswith(text_prefix):
                    continue
                print(file)

                file_filepath = folder_path + "/" + dir + "/" + file
                text = join_files([file_filepath])
                year = os.path.basename(folder_path).split("-")[3]
                journal = dir
                page_src = file
                summary = ""
                summary_path = folder_path + "/" + dir + f"/{summary_prefix}" + file
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding="utf-8") as file:
                        summary = file.read()
                else:
                    #create empty summary
                    with open(summary_path, 'w', encoding="utf-8") as file:
                        file.write(f"{PROMPT}{text}")
                dataset_pair = create_dataset_pair(text, summary, year, journal, page_src, page_num)
                dataset_pair_list.append(dataset_pair)
                page_num += 1
                summary_total += " " + summary
                print("processed")
            summary_total = summary_total.replace("\n", " ")
            summary_total_path = folder_path + "/" + dir + f"/{SUMMARY_TOTAL}"
            summary_total_prompt_path = folder_path + "/" + dir + f"/{SUMMARY_TOTAL_PROMPT}"

            with open(summary_total_prompt_path, 'w', encoding="utf-8") as file:
                    file.write(f"{PROMPT}{summary_total}")

            if not os.path.exists(summary_total_path):
                with open(summary_total_path, 'w', encoding="utf-8") as file:
                    file.write(f"{PROMPT}{summary_total}")
            else:
                with open(summary_total_path, 'r', encoding="utf-8") as file:
                    summary_total = file.read()

            #data.append({dir:{"pages": dataset_pair_list, "summary_total": summary_total}})
            data_poc[dir] = {"pages": dataset_pair_list, "summary_total": summary_total}
            #text = join_files(txt_files_path)
            data_poc_p.extend(dataset_pair_list)
    return data_poc, data_poc_p
            
            

def create_dataset_pair(text, summary, year, issue, page_src, page_num):
    return {"text": text, "summary": summary, "year": year, "issue": issue, "page_src": page_src, "page_num": page_num}

#argument: a list of text files (their paths) result: their content joined together
def join_files(files):
    joined_text = ""
    for file in files:
        with open(file, 'r', encoding="utf-8") as file:
            joined_text += file.read()
    return joined_text

def create_dataset(dir_location, text_prefix="posel-od-cerchova-", folder_suffix="-ukazka", summary_prefix="summary-"):
    """
    Create a POC dataset and POC_P dataset from the specified directory location.

    Args:
        dir_location (str): The directory location where the dataset is stored.
        text_prefix (str, optional): The prefix for the text files. Defaults to "posel-od-cerchova-".
        folder_suffix (str, optional): The suffix for the folders. Defaults to "-ukazka".
        summary_prefix (str, optional): The prefix for the summary files. Defaults to "summary-".

    Returns:
        tuple: A tuple containing two elements:
            - dataset_poc (dict): POC dataset - Dictionary containing journals. These contain issues, which contain pages and the summary total.
            - dataset_poc_p (list): POC_P dataset - A list containing only pages.
    """
    
    dataset_poc = {}
    dataset_poc_p = []
    dirs = os.listdir(dir_location)
    for dir in dirs:
        dir: str
        if os.path.isdir(dir_location + "/" + dir):
            if not dir.endswith(folder_suffix):
                continue
            print(dir)
            el, data_2 = process_folder(dir_location + "/" + dir, text_prefix, summary_prefix)
            dataset_poc[dir] = el
            dataset_poc_p.extend(data_2)
    return dataset_poc, dataset_poc_p

def main():
    dataset, dataset2 = create_dataset(TEXT_LOC, TEXT_PREFIX, FOLDER_SUFFIX, SUMMARY_PREFIX)
    print(len(dataset2))
    dataset_poc_json = json.dumps(dataset, ensure_ascii=False)
    dataset_poc_p_json = json.dumps(dataset2, ensure_ascii=False)
    #export to json
    with open(DATASET_POC, 'w', encoding="utf-8") as file:
        file.write(dataset_poc_json)

    with open(DATASET_POC_P, 'w', encoding="utf-8") as file:
        file.write(dataset_poc_p_json)
main()