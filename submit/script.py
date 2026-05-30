import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    # 데이콘 서버 규정에 맞춘 철저한 상대경로 세팅
    test_path = os.path.join('data', 'test.csv')
    model_path = os.path.join('model')
    output_path = os.path.join('output', 'submission.csv')
    
    # 만약을 대비해 저장 폴더 사전 생성
    os.makedirs('output', exist_ok=True)
    
    # 1. 테스트 데이터 로드
    if not os.path.exists(test_path):
        print(f"❌ 에러: {test_path} 주소에 데이터가 없습니다.")
        return
    test_df = pd.read_csv(test_path)
    
    # 2. 학습 때와 100% 동일한 전처리 파이프라인 가동 (리스트 컴프리헨션 + 정규식)
    test_df['clean_text'] = [f"{t} {f}" for t, f in zip(test_df['title'].fillna(''), test_df['full_text'].fillna(''))]
    pattern = re.compile(r'[^\w\s]')
    test_df['clean_text'] = [pattern.sub('', text)[:2500] for text in test_df['clean_text']]
    
    # 3. 내 손으로 학습시켰던 model 폴더에서 가중치와 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # 서버에 GPU가 있으면 가속, 없으면 CPU 구동 보장 코드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 4. 고속 추론 (Batch 단위가 아닌 개별 문장 동적 패딩)
    predictions = []
    with torch.no_grad():
        for text in test_df['clean_text']:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)
            
    # 5. 데이콘 양식에 맞춰 결과 저장
    submit_df = pd.DataFrame({
        'id': test_df['id'],
        'label': predictions
    })
    submit_df.to_csv(output_path, index=False)
    print(f"🏁 데이콘 서버 추론이 정상 종료되었습니다. 저장 완료 -> {output_path}")

if __name__ == "__main__":
    main()