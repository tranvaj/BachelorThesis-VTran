from datasets import load_dataset
import rouge_raw
from transformers import pipeline
import configparser

config = configparser.ConfigParser()
config.read('config.cfg')

DATASET_DIR = config['SumeCzech']['dataset_dir'] 
MODEL_CHECKPOINT = config['mT5']['summarizer'] 
TEST_FILE = config['SumeCzech']['test'] 

max_input_length = 512
max_target_length = 512
print("Evaluating...")

data_files = {"test":f"{DATASET_DIR}/{TEST_FILE}"}
sc_set = load_dataset("json", data_files=data_files)

summarizer_t5 = pipeline(task='summarization', model=MODEL_CHECKPOINT, device_map="auto")
articles_text  = sc_set["test"]["text"]
abstract = sc_set["test"]["abstract"]

max_words = 512

truncated_articles = []

for article in articles_text:
    words = article.split()
    truncated_article = ' '.join(words[:max_words])
    truncated_articles.append(truncated_article)


summary = summarizer_t5(truncated_articles, min_length=0, max_length=1024)

summary_texts = [item['summary_text'] for item in summary]

#write the summaries to a file
with open("summary.txt", "w", encoding="utf-8") as file:
    for summary_text in summary_texts:
        file.write(summary_text + "\n")

eval = rouge_raw.RougeRaw()
roguerawscore = eval.corpus(gold=abstract, system=summary_texts)

print("ROUGE-1 F: ", roguerawscore["1"].f)
print("ROUGE-1 P: ", roguerawscore["1"].p)
print("ROUGE-1 R: ", roguerawscore["1"].r)

print("ROUGE-2 F: ", roguerawscore["2"].f)
print("ROUGE-2 P: ", roguerawscore["2"].p)
print("ROUGE-2 R: ", roguerawscore["2"].r)

print("ROUGE-L F: ", roguerawscore["L"].f)
print("ROUGE-L P: ", roguerawscore["L"].p)
print("ROUGE-L R: ", roguerawscore["L"].r)

