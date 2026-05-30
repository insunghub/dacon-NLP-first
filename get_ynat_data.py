import os
import pandas as pd

print("📥 KLUE-YNAT (뉴스 제목 다중 분류) 데이터를 다운로드하는 중...")

# 공식 KLUE GitHub에서 YNAT 원본 데이터를 직접 가져옵니다.
train_url = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/ynat-v1.1/ynat-v1.1_train.json"

try:
    df = pd.read_json(train_url)
    print("✅ 원본 데이터 다운로드 완료!")
except Exception as e:
    print("⚠️ 네트워크 문제로 인해 연습용 다중 분류 데이터를 자체 생성합니다.")
    # 네트워크 에러 대비용 견고한 Fallback 데이터 생성
    import random
    samples = [
        ("정부 부동산 대책 추가 발표 결정", "정치"), ("코스피 금리 동결 소식에 상승 마감", "경제"),
        ("도심 주택가 화재 발생 인명 피해 없어", "사회"), ("이번 주말 가볼 만한 감성 미술전", "생활문화"),
        ("백악관 긴급 성명 발표 전 세계 주목", "세계"), ("최신 AI 자율주행 칩 탑재 스마트폰 출시", "IT과학"),
        ("지구촌 대축제 개막식 한국 선수단 입장", "스포츠")
    ]
    data = [{"guid": f"fallback_{i}", "title": samples[i%7][0], "label": samples[i%7][1]} for i in range(3500)]
    df = pd.DataFrame(data)

# 7개의 텍스트 라벨을 숫자로 매핑 (0 ~ 6)
df['clean_label'] = df['label'].str.replace('/', '').str.replace(' ', '')
label_map = {'정치': 0, '경제': 1, '사회': 2, '생활문화': 3, '세계': 4, 'IT과학': 5, '스포츠': 6}
df['label_num'] = df['clean_label'].map(label_map)

# 데이터 정제 및 변환
df = df.dropna(subset=['label_num'])
df = df.rename(columns={'guid': 'id', 'title': 'full_text'})
df['title'] = '[뉴스]'
df['label'] = df['label_num'].astype(int)

# 노트북 사양을 고려한 골디락스 규모 샘플링 (학습 3,000건 / 시험 500건)
train_df = df.sample(n=min(3000, len(df)), random_state=42)
test_df = df.drop(train_df.index).sample(n=min(500, len(df)-len(train_df)), random_state=42)

train_df = train_df[['id', 'title', 'full_text', 'label']]
test_df = test_df[['id', 'title', 'full_text']]

# 폴더에 안전하게 덮어쓰기
os.makedirs('data', exist_ok=True)
os.makedirs('submit/data', exist_ok=True)

train_df.to_csv('data/train.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('data/test.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('submit/data/test.csv', index=False, encoding='utf-8-sig')

print("\n✅ 다중 분류 베이스라인 데이터 셋업 완료!")
print(f"📊 학습 데이터 (train.csv): {len(train_df)}행 (정답 클래스: 0 ~ 6)")
print(f"📊 시험 데이터 (test.csv): {len(test_df)}행")
print("👉 [라벨 정보] 0:정치, 1:경제, 2:사회, 3:생활문화, 4:세계, 5:IT과학, 6:스포츠")