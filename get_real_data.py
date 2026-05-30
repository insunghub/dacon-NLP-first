import os
import pandas as pd

print("📥 실제 대회 규모의 데이터(NSMC)를 GitHub에서 직접 다운로드하는 중...")

# 허깅페이스 채널 대신, 네이버 공식 NSMC GitHub 원본 주소에서 직접 데이터를 읽어옵니다.
train_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
test_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"

# 원본 파일이 탭(\t)으로 구분되어 있어 sep='\t'를 사용합니다.
raw_train = pd.read_csv(train_url, sep='\t')
raw_test = pd.read_csv(test_url, sep='\t')

# 노트북 CPU 사양을 고려해 실전 느낌이 나면서도 가벼운 사이즈로 샘플링 (3,000건 / 500건)
train_sampled = raw_train.sample(n=3000, random_state=42)
test_sampled = raw_test.sample(n=500, random_state=42)

# 우리 코드 구조(id, title, full_text, label)에 맞게 컬럼명 변경 및 가상 타이틀 추가
train_sampled = train_sampled.rename(columns={'document': 'full_text'})
train_sampled['title'] = '[실전 리뷰]'
train_df = train_sampled[['id', 'title', 'full_text', 'label']]

test_sampled = test_sampled.rename(columns={'document': 'full_text'})
test_sampled['title'] = '[실전 리뷰]'
test_df = test_sampled[['id', 'title', 'full_text']]

# 혹시 모를 빈 텍스트(결측치) 행 제거
train_df = train_df.dropna(subset=['full_text'])
test_df = test_df.dropna(subset=['full_text'])

# 기존 data 폴더 및 submit/data 폴더에 안전하게 분할 저장
os.makedirs('data', exist_ok=True)
os.makedirs('submit/data', exist_ok=True)

train_df.to_csv('data/train.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('data/test.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('submit/data/test.csv', index=False, encoding='utf-8-sig')

print("\n✅ 실제 대회급 데이터셋 구축 완벽 성공!")
print(f"📊 학습 데이터 (train.csv): {len(train_df)}행")
print(f"📊 시험 데이터 (test.csv): {len(test_df)}행")
print("👉 이제 안심하고 train_fast_baseline.py를 실행하셔도 됩니다!")