import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from sklearn.metrics import f1_score, accuracy_score

# 1. 데이터 로드
print("📦 데이터를 로드하는 중...")
train_df = pd.read_csv("/train.csv")

# 2. 가장 안정적인 KoELECTRA-v3 경로
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. 토큰화
def tokenize_function(examples):
    return tokenizer(examples["full_text"], padding="max_length", truncation=True, max_length=512)

dataset = Dataset.from_pandas(train_df)
dataset = dataset.map(tokenize_function, batched=True)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 4. 평가지표
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"macro_f1": f1_score(labels, preds, average='macro'), "accuracy": accuracy_score(labels, preds)}

# 5. 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=7)

# 6. 학습 설정
training_args = TrainingArguments(
    output_dir="./koelectra_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=True,
)

# 7. 트레이너 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("🚀 KoELECTRA 학습 시작!")
trainer.train()