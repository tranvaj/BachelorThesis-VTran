from datasets import load_dataset
import transformers
import json
import numpy as np

from transformers import Seq2SeqTrainingArguments
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer
from huggingface_hub import login
import evaluate
import nltk

nltk.download('punkt')

rouge_score = evaluate.load("rouge")

#replace with path to your dataset
DATASET_PATH = "/storage/plzen1/home/nuva/sumeczech_dataset"
OUTPUT_PATH = "/storage/plzen1/home/nuva/bp/models"
API_TOKEN = "" #replace with your huggingface token if USE_HF is set to True
USE_HF = False #set to True if you want to push the model to huggingface hub
RESUME_TRAINING = False #set to True if you want to resume training from a checkpoint

def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset["dev"].shuffle(seed=seed).select(range(num_samples))
    print(sample)

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["abstract"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    #uses regular rouge instead of rougeraw

    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=False
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}



max_input_length = 512
max_target_length = 512
if USE_HF:
    login(token=API_TOKEN)

model_checkpoint = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)



#data_files = {"train":"/storage/plzen1/home/nuva/sumeczech_dataset/sumeczech-1.0-train.jsonl", "test":"/storage/plzen1/home/nuva/sumeczech_dataset/sumeczech-1.0-test.jsonl","dev":"/storage/plzen1/home/nuva/sumeczech_dataset/sumeczech-1.0-dev.jsonl"}
data_files = {"train":f"{DATASET_PATH}/sumeczech-1.0-train.jsonl", 
              "test":f"{DATASET_PATH}/sumeczech-1.0-test.jsonl",
              "dev":f"{DATASET_PATH}/sumeczech-1.0-dev.jsonl"}

sc_set = load_dataset("json", data_files=data_files)
sc_filtered_set = sc_set
tokenized_datasets = sc_filtered_set.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    sc_filtered_set["train"].column_names
)

batch_size = 8
num_train_epochs = 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
args = Seq2SeqTrainingArguments(
    output_dir=f"{OUTPUT_PATH}/{model_name}-finetuned-sumeczech",
    evaluation_strategy="epoch",
    learning_rate=0.001,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=USE_HF,
)


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train(resume_from_checkpoint=RESUME_TRAINING) #remove resume_from_checkpoint=True if you want to train from scratch
print(trainer.evaluate())
if USE_HF:
    trainer.push_to_hub(commit_message="Training complete", tags="summarization") 

