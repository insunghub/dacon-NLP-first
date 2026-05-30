import os
import pandas as pd

# 윈도우 환경에 맞는 폴더 생성
os.makedirs('data', exist_ok=True)
os.makedirs('submit/model', exist_ok=True)

# 1. 가짜 학습 데이터 (텍스트 + 정답 레이블)
train_data = {
    'id': [1, 2, 3, 4, 5],
    'title': ['API 호출 오류 발생', '네이버 실시간 뉴스 크롤링 요청', '사용자 친화적 인사말 출력', 'DB 유저 테이블 쿼리 실패', '구글 날씨 정보 검색'],
    'full_text': ['서버에서 500 에러 코드가 반환되어 외부 API 연동에 실패했습니다.', 
                 '에이전트가 지정된 URL에 접속하여 헤드라인 텍스트 데이터를 긁어왔습니다.', 
                 '안녕하세요 주인님! 오늘도 좋은 하루 보내세요.', 
                 'SQL 구문 오류가 발견되어 유저 데이터를 로드하지 못했습니다.', 
                 '구글 날씨 API를 호출하여 내일의 서울 최고 기온 데이터를 수집했습니다.'],
    'label': [0, 1, 2, 0, 1]  # 0: 오류 대응, 1: 정보 검색, 2: 대화
}

# 2. 가짜 테스트 데이터 (정답 레이블 없음 - 채점용)
test_data = {
    'id': [6, 7],
    'title': ['네트워크 타임아웃 에러', '챗봇 환영 멘트'],
    'full_text': ['인증 토큰이 만료되어 외부 세션 접속이 끊어졌습니다.', '반갑습니다 에이전트 서비스입니다. 무엇을 도와드릴까요?']
}

# 파일 저장 (utf-8-sig로 저장해야 엑셀이나 한글이 깨지지 않습니다)
pd.DataFrame(train_data).to_csv('data/train.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(test_data).to_csv('data/test.csv', index=False, encoding='utf-8-sig')

print("✅ 연습용 가짜 데이터(train.csv, test.csv) 생성 완료!")