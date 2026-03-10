import pandas as pd

# 1. 데이터 불러오기
df = pd.read_csv('credit_risk_dataset.csv')

print("========== [ 1. 데이터 미리보기 (head) ] ==========")
print(df.head())
print("\n")

print("========== [ 2. 데이터 기본 정보 (info) ] ==========")
df.info()
print("\n")

print("========== [ 3. 데이터 타입 (dtypes) ] ==========")
print(df.dtypes)
print("\n")
