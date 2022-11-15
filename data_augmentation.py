import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor

import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#------------------------------------------------------------------ 학습데이터셋
from sklearn.model_selection import train_test_split

#------------------------------------------------------------------ CART(Classification and Regression Tree)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVC, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,  AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet  #----------------*****
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix


#------------------------------------------------------------------ 증강학습
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
from imblearn.over_sampling import SMOTE

#------------------------------------------------------------------ 스케일링
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


#------------------------------------------------------------------ 평가관련
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# MAE
# MSE  : (squared=True)
# RMSE : (squared=False)
# MSLE


# 모델
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier,RandomForestRegressor
from lightgbm import LGBMClassifier
from tqdm import tqdm


#------------------------------------------------------------------ 튜닝관련

import warnings
warnings.filterwarnings(action='ignore')
sns.set()


df= pd.read_csv('./data/real_final_data_for_modeling.csv')
# selected feature df 불러오기
feat_df = pd.read_csv('./data/feature_selection_df.csv')



# selected_x_data column 반환
def return_selected_data(df_hj, model, feature_list_name):
    df = df_hj.copy()
    for i in df.columns[1:]:
        df[i] = df[i].apply(lambda x: x[2:-2].replace("'", '').split(', '))
    tmp = df[df['model'] == model][feature_list_name].iloc[0]
    return tmp





# Build Model
# 학습한 모델 반환
def build_model(model, X, y):
    clf = model
    clf.fit(X, y)
    return clf


# 모델 검증
def evaluate_model(clf, X, y):
    # pred = clf.predict(X) # predicted classes
    # accuracy = accuracy_score(pred, y) # calculate accuracy
    # fpr, tpr, _ = roc_curve(y, clf.predict_proba(X)[:,1]) # roc_curve
    # auc_value = auc(fpr, tpr) # auc_value
    # report = classification_report(y, pred, labels=[0, 1], output_dict = True)
    # report_df = pd.DataFrame(report).transpose()
    # report_df = report_df.reset_index()
    # model_eval = report_df[report_df['index'].str.contains('1')][['precision', 'recall', 'f1-score']]
    # model_eval['accuracy'] = list(report_df[report_df['index'].str.contains('accuracy')]['support'])
    # model_eval['ROC'] = auc_value
    # cf_matrix = confusion_matrix(y, pred)

    pred = clf.predict(X)

    score_list = []
    Accuracy = round(accuracy_score(y, pred), 4)
    Precision = round(recall_score(y, pred, average='macro'), 4)
    F1 = round(f1_score(y, pred, average='macro'), 4)
    Recall = round(recall_score(y, pred, average='macro'), 4)

    score_list.append(Accuracy)
    score_list.append(Precision)
    score_list.append(F1)
    score_list.append(Recall)
    model_eval = pd.DataFrame([score_list], columns=['Accuracy', 'Precision', 'F1', 'Recall'])

    cf_matrix = confusion_matrix(y, pred)

    return model_eval, cf_matrix


# 모델 성능 저장
def model_eval_data(clf, X_train, y_train,
                    X_test, y_test,
                    model_eval_train,
                    model_eval_test,
                    Name=None):
    temp_eval_train, cf_matrix_train = evaluate_model(clf, X_train, y_train)
    temp_eval_test, cf_matrix_test = evaluate_model(clf, X_test, y_test)
    temp_eval_train.index = [Name]
    temp_eval_test.index = [Name]

    try:
        model_eval_train = model_eval_train.append(temp_eval_train)
        model_eval_test = model_eval_test.append(temp_eval_test)
    except:
        model_eval_train = temp_eval_train
        model_eval_test = temp_eval_test

    return model_eval_train, model_eval_test, cf_matrix_train, cf_matrix_test


# 초기화
# model_eval_train = pd.DataFrame({}, [])
# model_eval_test = pd.DataFrame({}, [])
model_eval_train = pd.DataFrame()
model_eval_test = pd.DataFrame()










# 1. SMOTE
df= pd.read_csv('./data/real_final_data_for_modeling.csv')


# smote 적용
def smote_eval(model, x_train, x_test, y_train, y_test, select_method):
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=1123)
    X_train_over, y_train_over = smote.fit_resample(x_train, y_train)
    print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트:', x_train.shape, y_train.shape)
    print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트:', X_train_over.shape, y_train_over.shape)
    print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())

    clf = build_model(model, X_train_over, y_train_over)  # 모델 학습

    # pred = model.predict(x_test)
    # pred_proba = model.predict_proba(x_test)
    # # cross_val_score(model, X_train_sm, y_train_sm, cv=skf, scoring='roc_auc').mean()

    # print('Confusion matrix')
    # print(confusion_matrix(y_test, pred))

    # print(f"Precision : {precision_score(y_test, pred, average='macro'):.4f}")
    # print(f"Recall    : {recall_score(y_test, pred, average='macro'):.4f}")
    # print(f"F1        : {f1_score(y_test, pred, average='macro'):.4f}")
    # print(f"Accuracy  : {accuracy_score(y_test, pred):.4f}")

    return model_eval_data(clf, X_train_over, y_train_over, x_test, y_test, model_eval_train,
                           model_eval_test, Name=f'SMOTE data - {model} & {select_method}')






# 1.2. SMOTE 모든 모델 & selected columns
# feature selection 이후 파트 다시 정리하기


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

for select_method in feat_df.columns[1:]:

    for model_name, model in models:
        # 데이터 초기화
        y_data = df['관리종목여부']
        x_data = df.drop(["관리종목여부"], axis=1)

        x_data_col = return_selected_data(feat_df, model_name, select_method)  # 모델별 feature select 방식별 선정된 feature 반환
        # data augmentation 이전에 미리 원본 데이터로 테스트 데이터셋 구축(stratify 기능을 통해 균형 있게 테스트셋 구축)

        # feautre 선별
        new_x_data = x_data[x_data_col]
        new_y_data = y_data

        # stratify를 통해 test 검증셋과 증강할 데이터셋 구분
        x_train, x_test, y_train, y_test = train_test_split(new_x_data, new_y_data, test_size=0.2, random_state=1123,
                                                            stratify=y_data)  # 데이터를 균등하게 나누고 test 검증하기 위해

        model_eval_train, model_eval_test, cf_matrix_train, cf_matrix_test = smote_eval(model, x_train, x_test, y_train,
                                                                                        y_test, select_method)



print(model_eval_test)