import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

def run_eda(file_path):
    print(f"--- 데이터 탐색형 데이터 분석 (EDA) 시작: {file_path} ---")
    
    # 1. 데이터 로드
    if not os.path.exists(file_path):
        print(f"오류: {file_path} 파일을 찾을 수 없습니다.")
        return
        
    df = pd.read_csv(file_path)
    
    # 2. 기본 정보 확인
    print("\n[1] 데이터의 기본 크기 (행, 열):")
    print(df.shape)
    
    print("\n[2] 데이터 정보 (결측치 및 데이터 타입):")
    print(df.info())
    
    print("\n[3] 수치형 변수 기초 통계량:")
    print(df.describe())
    
    print("\n[4] 결측치 개수:")
    print(df.isnull().sum())
    
    # ======= 시각화 =======
    print("\n시각화를 생성하고 있습니다. 창을 닫으면 다음으로 넘어갑니다...")
    
    # 시각화 1: 핵심 타겟 변수 (loan_status) 분포 확인
    plt.figure(figsize=(8, 5))
    sns.countplot(x='loan_status', data=df, palette='Set2')
    plt.title('대출 상태 분포 (0: 정상, 1: 채무 불이행)')
    plt.xlabel('대출 상태')
    plt.ylabel('신청 건수')
    plt.show()
    
    # 시각화 2: 나이 분포 확인 (이상치 확인 목적)
    plt.figure(figsize=(10, 5))
    sns.histplot(df['person_age'], bins=30, kde=True, color='skyblue')
    plt.title('대출 신청자 나이 분포')
    plt.xlabel('나이')
    plt.ylabel('빈도 수')
    plt.show()
    
    # 시각화 3: 대출 목적에 따른 대출 건수
    plt.figure(figsize=(12, 6))
    sns.countplot(y='loan_intent', data=df, order=df['loan_intent'].value_counts().index, palette='viridis')
    plt.title('대출 목적별 건수')
    plt.xlabel('건수')
    plt.ylabel('대출 목적')
    plt.show()
    
    # 시각화 4: 대출 등급(loan_grade)과 대출 상태(loan_status)의 관계
    plt.figure(figsize=(10, 6))
    sns.countplot(x='loan_grade', hue='loan_status', data=df, order=sorted(df['loan_grade'].dropna().unique()), palette='pastel')
    plt.title('대출 등급별 채무 불이행 비율')
    plt.xlabel('대출 등급 (A가 제일 좋음)')
    plt.ylabel('건수')
    plt.legend(title='대출 상태', labels=['정상 (0)', '채무 불이행 (1)'])
    plt.show()

    # 시각화 5: 수치형 변수 간의 상관관계 히트맵 (Target과의 관계 확인)
    plt.figure(figsize=(12, 10))
    # 수치형 컬럼만 선택
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_df.corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('수치형 변수 간의 상관관계 (Correlation Heatmap)')
    plt.show()
    
    print("\n--- EDA 완료 ---")

if __name__ == "__main__":
    run_eda("credit_risk_dataset.csv")
