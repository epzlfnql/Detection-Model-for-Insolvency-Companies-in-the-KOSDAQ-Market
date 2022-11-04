import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
# bining
# label encoder

#
df = pd.read_csv('./data/REALREALFINALJAM.csv')
def make_data_for_modeling(df):
    # 종목카드 6자리로 채워주기
    df['Stock'] = df['Stock'].astype('int')
    df['Stock'] = df['Stock'].astype('str')
    df['Stock'] = df['Stock'].apply(lambda x: x.zfill(6) if len(x)!=6 else 0)



    # 비재무 데이터 따로 뽑아내기
    no_financial_df = df[['자본잠식', '벌금', '불성실공시', '소송', '영업정지', '특허', '투자주의환기종목', '업종']]

    df_base = df[['Stock']]
    df_2018 = pd.concat([df_base, df[[col for col in df.columns if '2018' in col]]], axis=1)
    df_2018 = pd.concat([df_2018, no_financial_df], axis=1)

    df_2019 = pd.concat([df_base, df[[col for col in df.columns if '2019' in col]]], axis=1)
    df_2019 = pd.concat([df_2019, no_financial_df], axis=1)

    df_2020 = pd.concat([df_base, df[[col for col in df.columns if '2020' in col]]], axis=1)
    df_2020 = pd.concat([df_2020, no_financial_df], axis=1)

    df_2021 = pd.concat([df_base, df[[col for col in df.columns if '2021' in col]]], axis=1)
    df_2021 = pd.concat([df_2021, no_financial_df], axis=1)



    col_name = ['Stock', '총자산증가율', '매출액증가율', '총자본영업이익율', '매출액영업이익율', '자기자본비율', '유보액/납입자본비율', '총자본회전율', '영업활동으로인한현금흐름',
                '관리종목여부', 'Z_score', '부실기업판별_알Z', '부도확률_O', 'F_score', 'K_score', '부실여부판단_K', '성장성', ' 수익성', '안정성',
                '활동성', '현금흐름', '자본잠식', '벌금', '불성실공시', '소송', '영업정지', '특허', '투자주의환기종목', '업종']

    final_df = pd.DataFrame(columns=col_name)

    # Concat을 위해dataframe 컬럼명 변경
    df_2018.columns = col_name
    df_2019.columns = col_name
    df_2020.columns = col_name
    df_2021.columns = col_name

    final_df = pd.concat([final_df, df_2018], axis=0)
    final_df = pd.concat([final_df, df_2019], axis=0)
    final_df = pd.concat([final_df, df_2020], axis=0)
    final_df = pd.concat([final_df, df_2021], axis=0).reset_index(drop=True)


    # Z_score 채우기
    final_df['Z_score'].fillna(0, inplace=True)


    # 문자열 수치형 변환
    for i in final_df.columns[1:]:
        final_df[i] = final_df[i].astype('float')

    # 타겟값 맨마지막으로 빼주기
    y = final_df['관리종목여부'].values
    final_df.drop('관리종목여부',axis=1, inplace=True)
    final_df['관리종목여부'] = y


    return final_df



df = make_data_for_modeling(df)
print(df.info())