import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 💡 [보너스] 지저분한 허깅페이스 경고창 터미널에서 안 보이게 가리기
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 모델 이름과 토크나이저 설정
model_name = "monologg/koelectra-small-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples['clean_text'], truncation=True, max_length=128)

if __name__ == '__main__':
    
    # 1. 데이터 로드
    train_df = pd.read_csv('data/train.csv')

    # 리스트 컴프리헨션 고속 문자열 전처리
    train_df['clean_text'] = [f"{t} {f}" for t, f in zip(train_df['title'].fillna(''), train_df['full_text'].fillna(''))]
    pattern = re.compile(r'[^\w\s]')
    train_df['clean_text'] = [pattern.sub('', text)[:2500] for text in train_df['clean_text']]

    # 허깅페이스 데이터셋 변환
    hf_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])

    # CPU 멀티프로세싱 병렬 토큰화
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, num_proc=2)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # 3개의 클래스를 분류하는 AI 모델 정의
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 초고속 연습용 학습 파라미터 세팅
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10,
        report_to="none"
    )

    # 💡 [핵심 수정] 최신 Transformers 버전의 문법에 맞춰 
    # tokenizer=tokenizer 파라미터를 processing_class=tokenizer 로 변경했습니다.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer  
    )

    print("🚀 모든 환경 설정 통과! 가짜 데이터로 모델 학습을 시작합니다...")
    trainer.train()

    # 학습 완료 후 모델 저장
    save_path = os.path.join('submit', 'model')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"🎉 모델이 안전하게 상대경로 폴더에 저장되었습니다 -> {save_path}")