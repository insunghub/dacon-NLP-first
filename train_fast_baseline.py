import os
import re
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
# 데이콘 평가지표 계산을 위해 sklearn 라이브러리를 사용합니다.
from sklearn.metrics import f1_score, accuracy_score

# 1. 시스템 경고창 숨기기
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. 가성비 최고 성능의 한국어 모델 및 토크나이저 바깥 선언 (윈도우 멀티프로세싱 에러 방지)
model_name = "monologg/koelectra-small-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples['clean_text'], truncation=True, max_length=128)

# 💡 [실전 무기] 데이콘 채점 기준(Macro F1)과 정확도를 실시간 계산하는 함수
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    # 데이콘 단골 평가지표: Macro F1-Score
    macro_f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    
    return {"macro_f1": macro_f1, "accuracy": acc}


if __name__ == '__main__':
    print("🔮 실전 압축형 베이스라인 가동 시작...")
    
    # 3. 데이터 로드 (실제 대회 데이터가 들어와도 그대로 작동합니다)
    train_df = pd.read_csv('data/train.csv')

    # 고속 전처리 (텍스트 합치기 및 특수문자 제거)
    train_df['clean_text'] = [f"{t} {f}" for t, f in zip(train_df['title'].fillna(''), train_df['full_text'].fillna(''))]
    pattern = re.compile(r'[^\w\s]')
    train_df['clean_text'] = [pattern.sub('', text)[:2500] for text in train_df['clean_text']]

    # 4. 판다스를 허깅페이스 데이터셋으로 변환
    hf_dataset = Dataset.from_pandas(train_df[['clean_text', 'label']])

    # 🌟 [실전 무기] 데이터를 학습용(80%)과 검증용(20%)으로 자동 분리
    # 컴퓨터가 자체 모의고사를 볼 수 있게 방을 나눠주는 것입니다.
    dataset_split = hf_dataset.train_test_split(test_size=0.2, seed=42)
    train_data = dataset_split['train']
    val_data = dataset_split['test']

    # 병렬 토큰화 진행
    train_tokenized = train_data.map(tokenize_function, batched=True, num_proc=2)
    val_tokenized = val_data.map(tokenize_function, batched=True, num_proc=2)

    # 정답 컬럼 이름 변경 (label -> labels)
    train_tokenized = train_tokenized.rename_column("label", "labels")
    val_tokenized = val_tokenized.rename_column("label", "labels")

    # 5. 모델 정의 (클래스 3개 분류)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)

    # 6. 실전형 초고속 학습 파라미터 세팅
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,                  # 3번만 굵고 짧게 교과서 정독
        per_device_train_batch_size=8,       # 배치 사이즈를 8로 키워 학습 속도 대폭 상승
        per_device_eval_batch_size=8,
        
        # 🌟 모의고사(Evaluation) 전략 설정
        eval_strategy="epoch",               # 매 에포크(공부 1턴)가 끝날 때마다 시험 보기
        save_strategy="epoch",               # 매 에포크마다 모델 저장
        logging_steps=10,
        
        # 🌟 [실전 무기] 가장 성적이 좋았던 리즈시절 모델을 최종 선택
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",    # 최고 점수의 기준은 Macro F1 점수!
        greater_is_better=True,
        
        report_to="none"
    )

    # 7. 트레이너 가동 (검증 데이터와 평가지표 함수 추가)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,          # 👈 모의고사 시험지 주입
        processing_class=tokenizer,
        compute_metrics=compute_metrics      # 👈 채점관 주입
    )

    print("🚀 실전 압축형 학습을 시작합니다. 모의고사 점수를 확인하세요!")
    trainer.train()
    # 8. 가장 똑똑했던 상태의 모델을 최종 저장
    save_path = os.path.join('submit', 'model')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"\n🎉 [성공] 가장 점수가 높았던 최적의 모델이 저장되었습니다! -> {save_path}")