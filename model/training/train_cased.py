from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset
import numpy as np
import evaluate
from pathlib import Path

BATCH_SIZE = 16

dataset = load_dataset(
    "bigbio/bc5cdr",
    name="bc5cdr_bigbio_kb",
    trust_remote_code=True
)

label_list = ["O", "B-Disease", "I-Disease", "B-Chemical", "I-Chemical"]
label_encoding_dict = {label: i for i, label in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-cased"
)
data_collator = DataCollatorForTokenClassification(tokenizer)

model = AutoModelForTokenClassification.from_pretrained(
    "google-bert/bert-base-cased",
    attn_implementation="sdpa",
    num_labels=len(label_list),
    id2label=dict(enumerate(label_list)),
    label2id={label: index for index, label in enumerate(label_list)},
)

def tokenize_and_align_labels(examples):
    texts = []
    for passages in examples["passages"]:
        texts.append(" ".join([p["text"][0] for p in passages]))
        
    tokenized_inputs = tokenizer(
        texts, 
        truncation=True, 
        max_length=512, 
        return_offsets_mapping=True
    )

    labels = []
    
    for i, entities in enumerate(examples["entities"]):
        offsets = tokenized_inputs["offset_mapping"][i]
        
        label_ids = [0] * len(offsets)
        
        for entity in entities:
            ent_type = entity["type"] 
            ent_start, ent_end = entity["offsets"][0]
            
            if f"B-{ent_type}" not in label_encoding_dict:
                continue
                
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end:
                    continue
                    
                if tok_start >= ent_start and tok_end <= ent_end:
                    if tok_start == ent_start:
                        label_ids[idx] = label_encoding_dict[f"B-{ent_type}"]
                    else:
                        label_ids[idx] = label_encoding_dict[f"I-{ent_type}"]
                        
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                label_ids[idx] = -100
                
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping")
    
    return tokenized_inputs

train_tokenized_datasets = dataset["train"].map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = dataset["test"].map(tokenize_and_align_labels, batched=True)
validation_tokenized_datasets = dataset["validation"].map(tokenize_and_align_labels, batched=True)

metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
    
args = TrainingArguments(
    learning_rate=1e-4,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=1e-5,
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=validation_tokenized_datasets,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

save_dir = Path(__file__).resolve().parents[1] / "saved" / "c-ner.model"
save_dir.parent.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(save_dir))
tokenizer.save_pretrained(str(save_dir))