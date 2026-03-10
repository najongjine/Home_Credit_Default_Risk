import pandas as pd
from imblearn.over_sampling import SMOTE
import os

def create_oversampled_dataset(input_file, output_file):
    print(f"--- 데이터 로딩: {input_file} ---")
    
    if not os.path.exists(input_file):
        print(f"오류: {input_file} 파일을 찾을 수 없습니다.")
        return
        
    df = pd.read_csv(input_file)
    
    print("\n[원본 데이터 상태 (loan_status)]")
    status_counts = df['loan_status'].value_counts()
    print(f"정상 (0): {status_counts.get(0, 0):,}건")
    print(f"채무 불이행 (1): {status_counts.get(1, 0):,}건")
    
    # 1. 전처리: 결측치 처리 및 식별 불가형 데이터 변환
    print("\n--- 전처리 진행 중 ---")
    print("결측치를 중앙값으로 채우고, 문자열 데이터를 숫자형(One-Hot Encoding)으로 변환합니다.")
    
    # 결측치 채우기 (연속형은 중앙값, 범주형은 최빈값 등 상황에 맞게)
    # 여기서는 간단히 중앙값으로 대체합니다.
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    
    # 범주형 데이터(object 타입)를 숫자형으로 변환 (One-Hot Encoding)
    # SMOTE 알고리즘은 숫자형 데이터만 처리할 수 있기 때문입니다.
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # 특징(X)과 타겟(y) 분리
    X = df_encoded.drop('loan_status', axis=1)
    y = df_encoded['loan_status']
    
    # 2. SMOTE (오버샘플링) 적용
    print("\n--- SMOTE 적용 중 ---")
    print("소수 클래스(채무 불이행)의 가상 데이터를 생성하여 5:5 비율로 맞춥니다.")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("\n[SMOTE 적용 완료. 데이터 상태]")
    resampled_counts = y_resampled.value_counts()
    print(f"정상 (0): {resampled_counts.get(0, 0):,}건")
    print(f"채무 불이행 (1): {resampled_counts.get(1, 0):,}건")
    
    # 3. 새로운 데이터프레임으로 결합 후 저장
    print("\n--- 데이터를 CSV 파일로 저장 중 ---")
    
    # X_resampled와 y_resampled를 다시 하나의 DataFrame으로 합침
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    
    # 컬럼 순서를 원본 df_encoded와 동일하게 맞춰주기 (옵션)
    df_resampled = df_resampled[df_encoded.columns]
    
    # CSV로 저장
    df_resampled.to_csv(output_file, index=False)
    
    print(f"\n성공적으로 저장되었습니다! 파일명: {output_file}")
    print(f"전체 데이터 건수: {len(df_resampled):,}건")

if __name__ == "__main__":
    input_filename = "credit_risk_dataset.csv"
    output_filename = "credit_risk_dataset_v2.csv"
    
    create_oversampled_dataset(input_filename, output_filename)
