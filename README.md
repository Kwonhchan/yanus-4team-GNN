# solo-project
this repo is for solo-project

##### Frontend: React.js 
######    Version : nodejs 20.11.0
##### backend: SpringBoot
######    Version : java 11.0.15, spring-boot 2.7.5

# 추천 시스템 API 문서

## 개요

본 문서는 사용자 데이터를 분석하여 사용자의 선호도 및 상호 작용을 기반으로 아이템을 추천하는 고급 머신러닝 모델을 활용하는 추천 시스템 API의 사용 및 통합에 대한 상세 정보를 제공합니다. FastAPI를 기반으로 구축되어 고성능 및 쉬운 확장성을 보장하며, 전자상거래부터 콘텐츠 플랫폼에 이르기까지 다양한 애플리케이션에 적용될 수 있습니다.

## 시작하기

### 전제 조건

- Python 3.6 이상
- FastAPI
- Uvicorn
- PyTorch
- Pandas
- Scikit-learn

### 설치 및 실행

시스템 설정을 위해 필요한 라이브러리를 설치하려면 다음의 명령어를 실행하십시오.

```bash
pip install fastapi uvicorn torch pandas scikit-learn
```

## API 사용 방법

### 추천 요청

추천을 받기 위해서는 /recommend/ 경로로 POST 요청을 보내야 합니다. 요청 본문에는 사용자 데이터가 포함되어 있어야 합니다.

json 파일을 받아서 TRAVEL_ID,VISIT_AREA_NM,RESIDENCE_TIME_MIN,VISIT_AREA_TYPE_CD,DGSTFN,REVISIT_INTENTION,RCMDTN_INTENTION,COMBINED_ADDR,GENDER,AGE_GRP,FAMILY_MEMB,TRAVEL_COMPANIONS_NUM의 컬럼들에 해당되는 값을 모두 채워야 함.

### 입력부분의 수정이 필요함.

예시부분(csv파일로 받지 못할 경우, 이 형식으로 받을 수 있음)
```
# 웹에서 받은 데이터 예제
# 칼럼 수와 밑의 데이터 수가 일치해야함
# 아래 내용은 COMBINED_ADDR부분을 제거한 내용

raw_data = """TRAVEL_ID,VISIT_AREA_NM,RESIDENCE_TIME_MIN,DGSTFN,REVISIT_INTENTION,RCMDTN_INTENTION,GENDER,AGE_GRP,FAMILY_MEMB,TRAVEL_COMPANIONS_NUM
a_a003071,성산일출봉,80.0,11,5.0,5.0,5.0,여,30,2,1
a_a003071,올레길,60.0,11,5.0,5.0,5.0,여,30,2,1
"""

df1 = pd.read_csv(io.StringIO(raw_data))
graph_data = GraphDataU(df1)
```

위 코드 내용 처럼사용자데이터는 csv파일로 변환되어 모델에 입력되거나 csv 그 자체로 df를 선언해야함.

요청예시
```
curl -X 'POST' \
  'http://127.0.0.1:8000/recommend/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user_data":"user_data_csv_string"}'
```

이 요청은 사용자 데이터를 기반으로 시스템이 처리를 수행하고, 추천 아이템 목록을 반환합니다.

## 개발 및 테스트

API의 개발과 테스트를 위해 FastAPI는 자동으로 생성되는 문서를 제공합니다. 서버가 실행 중이면, 브라우저에서 http://127.0.0.1:8000/docs로 이동하여 API의 상세한 스펙과 인터페이스를 확인할 수 있습니다. 이 문서 페이지에서는 실제 API 요청을 시도하고 결과를 볼 수도 있습니다.
