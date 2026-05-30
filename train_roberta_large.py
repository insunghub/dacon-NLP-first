import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from transformers import EarlyStoppingCallback

# 1. 10GB 용량 제한을 뚫어낼 최강의 대형 모델 장착!
model_name = "klue/roberta-large"

# 2. 데이터 불러오기 (코랩 업로드 경로)
print("📊 데이터 로딩 중...")
train_df = pd.read_csv("/train.csv")

# 7개 다중 분류인지 확인하고 처리
num_labels = int(train_df['label'].nunique()) if 'label' in train_df.columns else 7
print(f"정답 클래스 개수: {num_labels}")

# 데이터 분할 (학습용 80%, 검증용 20%)
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

# Hugging Face 데이터셋 변환
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# 3. 토크나이저 및 모델 로드
print(f"📥 {model_name} 모델 다운로드 중... (약 1.3GB)")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 텍스트 토큰화 함수
def tokenize_function(examples):
    return tokenizer(examples['full_text'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 4. 평가 지표 정의 (정확도)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# 5. 하이퍼파라미터 설정 (GPU가 있으므로 안심하고 돌립니다)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-6,  # 대형 모델은 보통 2e-5 아랫선이 안정적입니다
    per_device_train_batch_size=8, # 대형 모델이라 메모리 폭발 방지용으로 8 지정
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True, # GPU 전용 연산 가속 치트키 (속도 2배 향상)
)

# 6. 트레이너 가동 및 초고속 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] # 👈 오차가 1번이라도 증가하면 정지
)

print("🚀 GPU 부스터 가동! 대형 모델 학습을 시작합니다.")
trainer.train()
print("✅ 학습 완료!")