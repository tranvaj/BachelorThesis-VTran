from datasets import load_dataset
import os
from pathlib import Path
import json

PROMPT = "Vytvoř shrnutí následujícího textu ve stylu novináře. Počet vět <= 5:\n"
SUMMARY_TOTAL = "summary-total.txt"
SUMMARY_TOTAL_PROMPT = "summary-total-prompt.txt"
DATASET_POC = "D:/FAVZCU/BP/summarization_bp/text/dataset_poc.json"
DATASET_POC_P = "D:/FAVZCU/BP/summarization_bp/text/dataset_poc_p.json"
TEXT_LOC = "D:/FAVZCU/BP/summarization_bp/text" #where posel od cerchova folders are located

def process_folder(folder_path, text_prefix, summary_prefix, max_dirs=-1):
    dirs = os.listdir(folder_path)
    data = {}
    data2 = []
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
            data[dir] = {"pages": dataset_pair_list, "summary_total": summary_total}
            #text = join_files(txt_files_path)
            data2.extend(dataset_pair_list)
    return data, data2
            
            

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
    dataset = {}
    dataset2 = []
    dirs = os.listdir(dir_location)
    for dir in dirs:
        dir: str
        if os.path.isdir(dir_location + "/" + dir):
            if not dir.endswith(folder_suffix):
                continue
            print(dir)
            el, data_2 = process_folder(dir_location + "/" + dir, text_prefix, summary_prefix)
            dataset[dir] = el
            dataset2.extend(data_2)
    return dataset, dataset2

def main():
    dataset, dataset2 = create_dataset(TEXT_LOC)
    print(len(dataset2))
    dataset_json = json.dumps(dataset, ensure_ascii=False)
    dataset2_json = json.dumps(dataset2, ensure_ascii=False)
    #export to json
    with open(DATASET_POC, 'w', encoding="utf-8") as file:
        file.write(dataset_json)

    with open(DATASET_POC_P, 'w', encoding="utf-8") as file:
        file.write(dataset2_json)
main()