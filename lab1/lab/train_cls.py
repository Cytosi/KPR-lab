import os
import argparse
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

from BERT import DistilBertForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, DownloadMode
import numpy as np

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean", "max"],
                    help="池化策略: cls, mean, max")
args = parser.parse_args()

print(f"导入完成... 使用池化策略: {args.pooling}")

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

############模型定义
tokenizer = AutoTokenizer.from_pretrained("./cache/distilbert")
model = DistilBertForSequenceClassification.from_pretrained(
    "./cache/distilbert", num_labels=2, id2label=id2label, label2id=label2id,
    pooling_strategy=args.pooling  # 传入池化策略
)

###########数据集准备
print("加载数据集...")
from datasets import Dataset, DatasetDict

cache_path = "./cache/stanfordnlp___parquet/stanfordnlp--sst2-c614fb49d6bf6d65/0.0.0/14a00e99c0d15a23649d0db8944380ac81082d4b021f398733dd84f3a6c569a7"
dataset = DatasetDict({
    "train": Dataset.from_file(f"{cache_path}/parquet-train.arrow"),
    "validation": Dataset.from_file(f"{cache_path}/parquet-validation.arrow"),
    "test": Dataset.from_file(f"{cache_path}/parquet-test.arrow"),
})
print("数据集加载完成...")

def tokenize_cls(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True)

print("处理数据集...")
tokenized_dataset = dataset.map(tokenize_cls, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("数据预处理完成...")

##########评测指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    print("\n"*3)
    print("metrics:")
    print(f"\nAccuracy: {accuracy:.4f}\n")
    print("\n"*3)
    return {"accuracy": accuracy}
    

###########训练参数
training_args = TrainingArguments(
    output_dir=f"./ckpt/CLS_ckpt_{args.pooling}",  # 不同池化策略保存到不同目录
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    warmup_ratio=0.1,
    fp16=True,
)

###############模型训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


print("\n" + "="*50)
print(f"池化策略: {args.pooling}")
print("="*50)
