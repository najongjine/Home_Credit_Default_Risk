import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def preprocess_step1(df):
    """
    LightGBM 최적화 5단계 전처리 파이프라인
    - 1단계: 결측치 및 이상치 파악 및 처리
    """
    print("========== [ 1단계: 결측치 및 이상치 처리 ] ==========")
    print(f"초기 데이터 형태: {df.shape}\n")
    
    df_clean = df.copy()
    
    # ---------------------------------------------------------
    # 1. 논리적 오류 (이상치) 처리
    # ---------------------------------------------------------
    # 1-1. 나이(person_age) 이상치: 100세 초과인 경우 (예: 144세 등)
    # 극단적인 이상치이므로 제거하는 것이 일반적입니다.
    age_outliers = df_clean[df_clean['person_age'] > 100]
    print(f"[이상치 제거] 나이가 100세를 초과하는 데이터 수: {len(age_outliers)}건")
    df_clean = df_clean[df_clean['person_age'] <= 100]
    
    # 1-2. 직장 경력(person_emp_length)이 나이(person_age)보다 긴 오류
    # 물리적으로 불가능한 데이터이므로 제거합니다. (또는 결측치로 만든 뒤 대체 가능)
    # 여기서는 직장 경력이 나이보다 크거나 같은 경우를 오류로 간주하여 제거합니다.
    # (예를 들어, 보통 15세 이전에는 합법적 근무가 어려울 수 있으나
    # 최소한 '경력 >= 나이'인 경우는 확실한 오류입니다.)
    invalid_emp = df_clean[df_clean['person_emp_length'] >= df_clean['person_age']]
    print(f"[이상치 제거] 직장 경력이 나이보다 길거나 같은 데이터 수: {len(invalid_emp)}건")
    df_clean = df_clean[~df_clean.index.isin(invalid_emp.index)]
    
    # ※ 추가적인 이상치 처리 (선택)
    # 직장 경력이 60년 이상인 경우 등 상식적이지 않은 값 확인 및 제거 가능
    emp_outliers = df_clean[df_clean['person_emp_length'] > 60]
    print(f"[이상치 제거] 직장 경력이 60년을 초과하는 데이터 수: {len(emp_outliers)}건")
    df_clean = df_clean[df_clean['person_emp_length'] <= 60]

    # ---------------------------------------------------------
    # 2. 결측치 (Missing Values) 처리
    # ---------------------------------------------------------
    # 데이터를 확인해보면 person_emp_length와 loan_int_rate에 결측치가 존재합니다.
    print(f"\n[결측치 확인 - 처리 전]\n{df_clean.isnull().sum()}\n")
    
    # LightGBM은 자체적으로 결측치를 처리할 수 있지만, 
    # 통계적/논리적으로 타당하다면 채워주는 것이 좋습니다.
    
    # 2-1. person_emp_length(근속 연수) 결측치 대체
    # 근속 연수는 소득이나 나이 등과 연관이 있지만, 일반적으로 값이 치우쳐 있어
    # 평균(mean)보다는 중앙값(median) 대체를 선호합니다.
    if df_clean['person_emp_length'].isnull().sum() > 0:
        emp_median = df_clean['person_emp_length'].median()
        df_clean['person_emp_length'] = df_clean['person_emp_length'].fillna(emp_median)
        print(f"[결측치 대체] 'person_emp_length' 결측치를 중앙값({emp_median})으로 대체")
        
    # 2-2. loan_int_rate(이자율) 결측치 대체
    # 이자율은 대출 등급(loan_grade)과 강력한 상관관계가 있으므로,
    # 각 대출 등급별 이자율의 중앙값으로 채워주는 것이 가장 이상적입니다.
    if df_clean['loan_int_rate'].isnull().sum() > 0:
        # 대출 등급별 이자율 중앙값 계산
        grade_median_rates = df_clean.groupby('loan_grade')['loan_int_rate'].median()
        
        # 결측치를 해당 대출 등급의 중앙값으로 대체
        def fill_int_rate(row):
            if pd.isna(row['loan_int_rate']):
                return grade_median_rates[row['loan_grade']]
            return row['loan_int_rate']
            
        df_clean['loan_int_rate'] = df_clean.apply(fill_int_rate, axis=1)
        print("[결측치 대체] 'loan_int_rate' 결측치를 각 대출 등급(loan_grade)의 평균/중앙값으로 대체")
        
    print(f"\n처리 완료 후 데이터 형태: {df_clean.shape}")
    print("======================================================")
    return df_clean

def preprocess_step2(df):
    """
    LightGBM 최적화 5단계 전처리 파이프라인
    - 2단계: 명목 데이터 인코딩 (Category 타입 변환)
    """
    print("\n========== [ 2단계: 명목 데이터 인코딩 ] ==========")
    df_encoded = df.copy()
    
    # 범주형 데이터 변수 목록
    # LightGBM은 Category 데이터 타입을 기본적으로 지원하므로,
    # One-Hot Encoding을 피하고 Category 타입으로 변환하면 메모리와 트리 분기 효율성이 높아집니다.
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    
    print(f"[타입 변환] 범주형 변수들 {categorical_cols}을(를) 'category' 타입으로 일괄 변환합니다.")
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype('category')
            
    print(f"\n처리 완료 후 데이터 형태: {df_encoded.shape}")
    print("======================================================")
    return df_encoded

def preprocess_step3(df):
    """
    LightGBM 최적화 5단계 전처리 파이프라인
    - 3단계: 파생 변수 생성 및 다중공선성 처리
    """
    print("\n========== [ 3단계: 파생 변수 및 다중공선성 처리 ] ==========")
    df_engineered = df.copy()
    
    # 3-1. 불필요한 원본 변수 제거 (다중공선성 처리)
    # loan_percent_income (소득 대비 대출 비율) = loan_amnt / person_income 
    # 이 파생 변수가 이미 존재하므로, 원본 변수인 대출 금액(loan_amnt)과 연소득(person_income)은 
    # 중복된 정보를 가지고 있습니다. 이를 제거하여 모델을 가볍게 만들고 해석을 용이하게 합니다.
    cols_to_drop = ['loan_amnt', 'person_income']
    cols_to_drop_existing = [col for col in cols_to_drop if col in df_engineered.columns]
    
    if cols_to_drop_existing:
        df_engineered = df_engineered.drop(columns=cols_to_drop_existing)
        print(f"[변수 제거] 중복 정보(다중공선성) 제거를 위해 변수 삭제: {cols_to_drop_existing}")
    
    print(f"\n처리 완료 후 데이터 형태: {df_engineered.shape}")
    print("======================================================")
    return df_engineered

def preprocess_step4(df):
    """
    LightGBM 최적화 5단계 전처리 파이프라인
    - 4단계: 모델 학습 및 하이퍼파라미터 튜닝 (L1/L2 정규화 및 GPU/CPU 폴백 포함)
    """
    print("\n========== [ 4단계: 모델 학습 및 하이퍼파라미터 튜닝 ] ==========")
    
    # 특성(X)과 타겟(y) 분리
    target_col = 'loan_status'
    if target_col not in df.columns:
        print(f"오류: 타겟 변수 '{target_col}'가 존재하지 않습니다.")
        return None
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 학습 세트와 테스트 세트 분리 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"학습 데이터 형태: {X_train.shape}, 테스트 데이터 형태: {X_test.shape}")
    
    # LightGBM 데이터셋 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 공통 하이퍼파라미터 (과적합 방지를 위한 L1(reg_alpha)/L2(reg_lambda) 정규화 포함)
    base_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'reg_alpha': 0.1,    # L1 정규화
        'reg_lambda': 0.1,   # L2 정규화
        'random_state': 42,
        'verbose': -1
    }
    
    # GPU 우선 시도
    gpu_params = base_params.copy()
    gpu_params['device'] = 'gpu'
    
    model = None
    try:
        print("[모델 학습] GPU 가속을 시도합니다...")
        model = lgb.train(
            params=gpu_params,
            train_set=train_data,
            num_boost_round=500,
            valid_sets=[train_data, test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        print("[모델 학습] GPU를 사용해 학습을 완료했습니다.")
    except Exception as e:
        print(f"[모델 학습] GPU 학습 실패. CPU로 다시 시도합니다.")
        print(f"(참고 메시지: {e})")
        cpu_params = base_params.copy()
        cpu_params['device'] = 'cpu'
        
        model = lgb.train(
            params=cpu_params,
            train_set=train_data,
            num_boost_round=500,
            valid_sets=[train_data, test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        print("[모델 학습] CPU를 사용해 학습을 완료했습니다.")
        
    # 예측 및 평가
    if model is not None:
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)
        
        print(f"\n[모델 평가 결과]")
        print(f"Accuracy (정확도): {accuracy:.4f}")
        print(f"AUC (ROC 기반): {auc:.4f}")
    
    print("======================================================")
    return model

if __name__ == "__main__":
    # 데이터 로드
    file_path = "credit_risk_dataset_v2.csv"
    try:
        df_origin = pd.read_csv(file_path)
        
        # 1단계 전처리 파이프라인 수행
        df_step1_done = preprocess_step1(df_origin)
        
        # 2단계 전처리 파이프라인 수행 (라벨 인코딩 / 범주형 타입 변환)
        df_step2_done = preprocess_step2(df_step1_done)
        
        # 3단계 전처리 파이프라인 수행 (파생 변수 및 다중공선성 처리)
        df_step3_done = preprocess_step3(df_step2_done)
        
        # 4단계 전처리 파이프라인 수행 (모델 학습 및 튜닝)
        model = preprocess_step4(df_step3_done)
        
        print("\n[전체 파이프라인 완료]")
        
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
