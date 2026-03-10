Credit Risk Dataset -
https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
This dataset contains columns simulating credit bureau data


* credit_risk_dataset.csv -

변수명 (Feature),설명,상세 의미
person_age,나이,대출 신청자의 나이입니다.
person_income,연소득,신청자가 1년 동안 버는 총소득입니다.
person_home_ownership,주거 형태,"집을 소유하고 있는지 여부입니다. (예: 자가 소유, 전/월세, 담보대출 낀 자가 등)"
person_emp_length,근속 연수,현재 직장에서 일한 기간(년 수)입니다. 직업의 안정성을 보여줍니다.

변수명 (Feature),설명,상세 의미
loan_intent,대출 목적,"돈을 빌리는 이유입니다. (예: 학자금, 의료비, 창업, 주택 구입 등)"
loan_grade,대출 등급,금융기관에서 평가한 대출 등급입니다. (보통 A등급이 제일 좋고 밑으로 갈수록 위험함)
loan_amnt,대출 금액,빌리고자 하는 총액수입니다.
loan_int_rate,이자율,대출에 적용되는 금리(%)입니다.
loan_percent_income,소득 대비 대출 비율,내 연소득에서 대출금이 차지하는 비율입니다. (대출금 / 연소득)

변수명 (Feature),설명,상세 의미
cb_person_default_on_file,과거 채무 불이행 기록,과거에 대출을 갚지 못해 연체나 부도를 낸 이력이 있는지 여부입니다.
cb_preson_cred_hist_length,신용 거래 이력 기간,신용카드 사용이나 대출 등 금융권과 신용 거래를 해온 총 기간입니다.

**loan_status,대출 상태 (핵심),"0은 정상 상환, 1은 채무 불이행(연체/부도)을 뜻합니다. 보통 데이터 분석 시 이 값을 맞추는 것이 목표가 됩니다."**

----------------------------

========== [ 1. 데이터 미리보기 (head) ] ==========
   person_age  person_income person_home_ownership  person_emp_length  ... loan_status loan_percent_income  cb_person_default_on_file  cb_person_cred_hist_length
0          22          59000                  RENT              123.0  ...           1                0.59                          Y                           3
1          21           9600                   OWN                5.0  ...           0                0.10                          N                           2
2          25           9600              MORTGAGE                1.0  ...           1                0.57                          N                           3
3          23          65500                  RENT                4.0  ...           1                0.53                          N                           2
4          24          54400                  RENT                8.0  ...           1                0.55                          Y                           4

[5 rows x 12 columns]


========== [ 2. 데이터 기본 정보 (info) ] ==========
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32581 entries, 0 to 32580
Data columns (total 12 columns):
 #   Column                      Non-Null Count  Dtype
---  ------                      --------------  -----
 0   person_age                  32581 non-null  int64
 1   person_income               32581 non-null  int64
 2   person_home_ownership       32581 non-null  object
 3   person_emp_length           31686 non-null  float64
 4   loan_intent                 32581 non-null  object
 5   loan_grade                  32581 non-null  object
 6   loan_amnt                   32581 non-null  int64
 7   loan_int_rate               29465 non-null  float64
 8   loan_status                 32581 non-null  int64
 9   loan_percent_income         32581 non-null  float64
 10  cb_person_default_on_file   32581 non-null  object
 11  cb_person_cred_hist_length  32581 non-null  int64
dtypes: float64(3), int64(5), object(4)
memory usage: 3.0+ MB


========== [ 3. 데이터 타입 (dtypes) ] ==========
person_age                      int64
person_income                   int64
person_home_ownership          object
person_emp_length             float64
loan_intent                    object
loan_grade                     object
loan_amnt                       int64
loan_int_rate                 float64
loan_status                     int64
loan_percent_income           float64
cb_person_default_on_file      object
cb_person_cred_hist_length      int64
dtype: object




💡 LightGBM 최적화 5단계 전처리 파이프라인
- 1단계: 결측치 및 이상치 파악 (가장 먼저 수행)

모델에 데이터를 넣거나 변환하기 전에 가장 먼저 해야 하는 작업입니다.

대출 데이터의 경우 나이가 144살이거나, 직장 경력이 나이보다 많은 논리적 오류(이상치)를 찾아내어 제거하거나 평균값 등으로 대체합니다.

- 2단계: 명목 데이터 인코딩 👉 [수정] 원핫 인코딩 대신 라벨 인코딩 / Category 타입 변환

중요: LightGBM은 트리(Tree) 기반 모델 중에서도 범주형 변수를 가장 잘 다루는 모델입니다. 원핫 인코딩을 하면 데이터의 차원이 불필요하게 늘어나고 0이 너무 많아져(Sparse) 오히려 학습 속도와 성능이 떨어집니다.

해결책: '자가', '전세', '월세' 같은 텍스트 데이터를 0, 1, 2 같은 단순 정수로 바꾸는 라벨 인코딩(Label Encoding)을 하거나, Pandas에서 데이터 타입을 category로 변환만 해두면 됩니다. LightGBM이 알아서 최적의 기준으로 분류합니다.

- 3단계: 파생 변수 생성 및 다중공선성 확인

선형 회귀 모델과 달리 트리 모델은 다중공선성에 매우 강하므로, 굳이 다중공선성이 높다고 해서 변수를 무조건 기계적으로 삭제할 필요는 없습니다.

다만, 대출금액과 연소득을 활용해 이미 소득 대비 대출 비율이라는 더 좋은 변수가 있다면 중복되는 원본 변수들을 정리해 주는 것이 모델을 가볍게 만들고 나중에 결과를 해석할 때 유리합니다.

- 4단계: 모델 학습 및 하이퍼파라미터 튜닝 (L1/L2 정규화 포함)

전처리가 끝난 데이터를 모델에 넣고 학습합니다. 이때 과적합(Overfitting)을 막기 위해 말씀하신 reg_alpha(L1), reg_lambda(L2) 값을 부여합니다.

장착하고 계신 RTX 3060 8GB의 GPU 가속 환경(device='gpu')을 활용하면, 이 단계에서 트리의 깊이(max_depth)나 학습률(learning_rate) 같은 다양한 파라미터 조합을 수십 번씩 반복 테스트하더라도 순식간에 최적의 값을 찾아낼 수 있습니다.

- 5단계: 학습 후 중요도 분석(SHAP) 및 변수 최종 제거

1차 학습 결과를 바탕으로 변수 중요도(Feature Importance)나 SHAP Value를 뽑아봅니다.

모델의 예측에 거의 기여하지 않는 하위 10~20%의 변수들을 과감히 쳐내고, 남은 '진짜 중요한 변수'들만 가지고 다시 4단계로 돌아가 최종 모델을 훈련시킵니다.