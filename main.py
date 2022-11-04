# 1) feature selection code 넣기
# 2) bining 한거로 모델링 vs 재무제표 데이터로만 모델링
# 3)

# 라이브러리 & 함수 import
from data_for_modeling import *

df = pd.read_csv('./data/REALREALFINALJAM.csv')
df = make_data_for_modeling(df)
print(df.head(5))