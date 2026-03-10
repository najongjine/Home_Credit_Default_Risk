import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def check_and_balance_data(file_path):
    print(f"--- 데이터 로딩 및 불균형 확인: {file_path} ---")
    
    if not os.path.exists(file_path):
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
        return
        
    df = pd.read_csv(file_path)
    
    # 1. 원본 데이터 불균형 확인
    status_counts = df['loan_status'].value_counts()
    print("\n[원본 데이터 상태 (loan_status)]")
    print(f"정상 (0): {status_counts.get(0, 0):,}건 ({status_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"채무 불이행 (1): {status_counts.get(1, 0):,}건 ({status_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # 시각화 1: 원본 분포
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.countplot(x='loan_status', data=df, palette='Set2', ax=axes[0])
    axes[0].set_title('오버샘플링 전: 대출 상태 분포')
    axes[0].set_xlabel('대출 상태 (0: 정상, 1: 채무 불이행)')
    axes[0].set_ylabel('건수')
    
    # ==========================================
    # 2. 전처리: 결측치 처리 및 범주형 데이터 변환 (SMOTE를 위해 필수)
    # ==========================================
    print("\n--- 전처리 진행 중 (결측치 채우기 및 인코딩) ---")
    # 결측치 채우기 (단순히 중앙값/최빈값 사용 - 실제 분석시 더 정교해야 함)
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    
    # 범주형 데이터(object 타입)를 숫자형으로 변환 (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # 특징(X)과 타겟(y) 분리
    X = df_encoded.drop('loan_status', axis=1)
    y = df_encoded['loan_status']
    
    # ==========================================
    # 3. SMOTE (오버샘플링) 적용
    # ==========================================
    print("\n--- SMOTE 적용 중 (부족한 채무 불이행 데이터 생성) ---")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # 오버샘플링 후 결과 확인
    resampled_counts = y_resampled.value_counts()
    print("\n[SMOTE 적용 후 데이터 상태]")
    print(f"정상 (0): {resampled_counts.get(0, 0):,}건 ({resampled_counts.get(0, 0)/len(y_resampled)*100:.1f}%)")
    print(f"채무 불이행 (1): {resampled_counts.get(1, 0):,}건 ({resampled_counts.get(1, 0)/len(y_resampled)*100:.1f}%)")
    
    # 시각화 2: SMOTE 적용 후 분포
    sns.countplot(x=y_resampled, palette='Set2', ax=axes[1])
    axes[1].set_title('오버샘플링 후 (SMOTE): 대출 상태 분포')
    axes[1].set_xlabel('대출 상태 (0: 정상, 1: 채무 불이행)')
    axes[1].set_ylabel('건수')
    
    plt.tight_layout()
    plt.show()
    
    print("\n--- 완료 ---")
    print("이제 기계가 '어떤 것이 채무 불이행인지' 배울 수 있도록")
    print("가상의 채무 불이행 데이터를 진짜 데이터와 비슷한 패턴으로 생성(복제)했습니다!")

if __name__ == "__main__":
    check_and_balance_data("credit_risk_dataset.csv")
