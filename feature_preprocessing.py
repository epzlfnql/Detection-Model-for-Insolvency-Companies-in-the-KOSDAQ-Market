'''
1) Feature selection

2)
'''
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, RFECV
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import eli5
from eli5.sklearn import PermutationImportance

import numpy as np
from sklearn.feature_selection import RFE, RFECV
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier,RandomForestRegressor
from lightgbm import LGBMClassifier
from tqdm import tqdm

# 데이터 불러오기
df = pd.read_csv('./data/real_final_data_for_modeling.csv')

# 데이터 나누기
y_data = df['관리종목여부'].values
x_data = df.drop(['Stock', '관리종목여부'], axis=1)

# 모델
cat_clf = CatBoostClassifier(random_state=1123)
lda_clf = LDA(n_components=1)
RF_clf = RandomForestClassifier(random_state=1123)
EX_clf = ExtraTreesClassifier(random_state=1123)
LGB_clf = LGBMClassifier(random_state=1123)
GB_clf = GradientBoostingClassifier(random_state=1123)

models = [("CF", cat_clf),
          ("LDA", lda_clf),
          ("RF", RF_clf),
          ("EX", EX_clf),
          ("LGB", LGB_clf),
          ('GB', GB_clf)
          ]






# 1. RFE
def rfe(model, x_data, y_data, select_feat_num=21):
    # define RFE
    rfe = RFE(estimator=model, n_features_to_select=select_feat_num)
    res = rfe.fit(x_data, y_data)
    selected_features = res.support_
    return list(x_data.columns[selected_features])

# rfe_features_list
rfe_features_list = []
for name, model in tqdm(models):
  y_data = df['관리종목여부'].values
  x_data = df.drop(['Stock', '관리종목여부'], axis=1)
  clf = model
  rfe_feature = rfe(clf, x_data, y_data, select_feat_num=21)
  rfe_features_list.append(rfe_feature)


# 2. rfecv
def rfecv(model, x_data, y_data):
    min_features_to_select = 10  # 최소한으로 선택할 변수 개수
    step = 1  # 매 단계마다 제거할 변수 개수
    selector = RFECV(clf, step=step, cv=10, scoring='f1_micro', min_features_to_select=min_features_to_select,
                     verbose=0)
    selector = selector.fit(x_data, y_data)

    return list(x_data.columns[selector.support_])

# rfecv_feature
rfecv_features_list = []
for name, model in tqdm(models):
  y_data = df['관리종목여부'].values
  x_data = df.drop(['Stock', '관리종목여부'], axis=1)
  clf = model
  rfecv_feature = rfecv(clf, x_data, y_data)
  rfecv_features_list.append(rfecv_feature)




# 3. PermutationImportance
def permutation_selection(model, x_data, y_data):
  perm = PermutationImportance(cat_clf, scoring='f1', random_state=1123).fit(x_data, y_data)
  # eli5.show_weights(perm, top=30, feature_names = x_data.columns.tolist())
  df_fi = pd.DataFrame(dict(feature_names=x_data.columns.tolist(),
                            feat_imp=perm.feature_importances_,
                            std=perm.feature_importances_std_,
                            ))
  df_fi2 = df_fi[df_fi['feat_imp']!=0]
  selected_feature = list(df_fi2['feature_names'])
  return selected_feature


# perm_features_list
perm_features_list = []
for name, model in tqdm(models):
  y_data = df['관리종목여부'].values
  x_data = df.drop(['Stock', '관리종목여부'], axis=1)
  clf = model
  perm_feature = permutation_selection(model, x_data, y_data)
  perm_features_list.append(perm_feature)



# 4. stepwise(단계적 선택법)
def stepwise_feature_selection(x_data, y_data, variables=x_data.columns.tolist()):
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")

    y = y_data  ## 반응 변수

    selected_variables = []  ## 선택된 변수들
    sl_enter = 0.05
    sl_remove = 0.05

    sv_per_step = []  ## 각 스텝별로 선택된 변수들
    adjusted_r_squared = []  ## 각 스텝별 수정된 결정계수
    steps = []  ## 스텝
    step = 0
    while len(variables) > 0:
        remainder = list(set(variables) - set(selected_variables))
        pval = pd.Series(index=remainder)  ## 변수의 p-value
        ## 기존에 포함된 변수와 새로운 변수 하나씩 돌아가면서
        ## 선형 모형을 적합한다.
        for col in remainder:
            X = x_data[selected_variables + [col]]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit(disp=0)
            pval[col] = model.pvalues[col]

        min_pval = pval.min()
        if min_pval < sl_enter:  ## 최소 p-value 값이 기준 값보다 작으면 포함
            selected_variables.append(pval.idxmin())
            ## 선택된 변수들에대해서
            ## 어떤 변수를 제거할지 고른다.
            while len(selected_variables) > 0:
                selected_X = x_data[selected_variables]
                selected_X = sm.add_constant(selected_X)
                selected_pval = sm.OLS(y, selected_X).fit(disp=0).pvalues[1:]  ## 절편항의 p-value는 뺀다
                max_pval = selected_pval.max()
                if max_pval >= sl_remove:  ## 최대 p-value값이 기준값보다 크거나 같으면 제외
                    remove_variable = selected_pval.idxmax()
                    selected_variables.remove(remove_variable)
                else:
                    break

            step += 1
            steps.append(step)
            adj_r_squared = sm.OLS(y, sm.add_constant(x_data[selected_variables])).fit(disp=0).rsquared_adj
            adjusted_r_squared.append(adj_r_squared)
            sv_per_step.append(selected_variables.copy())
        else:
            break

    return selected_variables


# stepwise_feature_list
stepwise_feature = stepwise_feature_selection(x_data, y_data, variables=x_data.columns.tolist() )
stepwise_feature_list = []
for i in range(len(models)):
  stepwise_feature_list.append(stepwise_feature)




# 5. Select K Best & ANOVA


from sklearn.feature_selection import SelectKBest, chi2, f_classif


def select_k_best(x_data, y_data, k=13):
    # 특징 선택을 위한 selectKBest를 진행, K는 특성 수
    # chi2 는 일반적으로 분류문제에서 사용
    selector = SelectKBest(score_func=f_classif, k=k)
    z = selector.fit_transform(x_data, y_data)

    filter = selector.get_support()
    features = x_data.columns

    return list(features[filter])


select_k_list = select_k_best(x_data, y_data, k=16)

# select_k_feature_list
select_k_feature_list = []
for i in range(len(models)):
  select_k_feature_list.append(select_k_list)



def make_feature_selection_df(rfe_features_list, rfecv_features_list,perm_features_list, stepwise_feature_list, select_k_feature_list):
  tmp = pd.DataFrame()

  tmp['model'] = ["CF","LDA","RF","EX","LGB",'GB']

  tmp['rfe_features_list'] = rfe_features_list
  tmp['rfecv_features_list'] = rfecv_features_list
  tmp['perm_features_list'] = perm_features_list
  tmp['stepwise_feature_list'] = stepwise_feature_list
  tmp['select_k_feature_list'] = select_k_feature_list
  return tmp





# 모델별 selected feature 모아놓은 df 반환

feature_selection_df = make_feature_selection_df(rfe_features_list, rfecv_features_list, perm_features_list, stepwise_feature_list, select_k_feature_list)
feature_selection_df.to_csv('./data/feature_selection_df.csv', index=False)
