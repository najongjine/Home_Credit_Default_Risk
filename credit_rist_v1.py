import pandas as pd
import numpy as np

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

if __name__ == "__main__":
    # 데이터 로드
    file_path = "credit_risk_dataset.csv"
    try:
        df_origin = pd.read_csv(file_path)
        
        # 1단계 전처리 파이프라인 수행
        df_step1_done = preprocess_step1(df_origin)
        
        print("\n[전처리 완료된 데이터 미리보기]")
        print(df_step1_done.head())
        
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
