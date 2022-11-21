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
# 이후에 gan쓸때 전역변수 선언을 위해 이름 변경
df.rename(columns={' 당기순이익(손실)':'당기순이익'}, inplace= True)
df.rename(columns={'유보액/납입자본비율':'유보액_납입자본비율'}, inplace =True)

# selected feature df 불러오기
feat_df = pd.read_csv('./data/feature_selection_df.csv')


# selected_x_data column 반환
def return_selected_data(df_hj, model, feature_list_name):
    df = df_hj.copy()
    for i in df.columns[1:]:
        df[i] = df[i].apply(lambda x: x[2:-2].replace("'", '').split(', '))
    tmp = df[df['model'] == model][feature_list_name].iloc[0]

    return tmp




# Build model
# 학습한 모델 반환
def build_model(model, X, y):
    clf = model
    clf.fit(X, y)
    return clf


# 모델 검증
def evaluate_model(clf, X, y):
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

# feature selection 이후 파트 다시 정리하기

y_data = df['관리종목여부']
x_data = df.drop(["관리종목여부"], axis=1)

# data augmentation 이전에 미리 원본 데이터로 테스트 데이터셋 구축(stratify 기능을 통해 균형 있게 테스트셋 구축)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1123, stratify=y_data) # 데이터를 균등하게 나누고 test 검증하기 위해

print("shape 확인")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print()
print(f'train 데이터 타겟값 분포 \n{pd.Series(y_train).value_counts()}')
print()
print(f'test 데이터 타겟값 분포 \n{pd.Series(y_test).value_counts()}')

# feature selection 이후 파트 다시 정리하기

y_data = df['관리종목여부']
x_data = df.drop(["관리종목여부"], axis=1)

# data augmentation 이전에 미리 원본 데이터로 테스트 데이터셋 구축(stratify 기능을 통해 균형 있게 테스트셋 구축)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1123,
                                                    stratify=y_data)  # 데이터를 균등하게 나누고 test 검증하기 위해


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
                           model_eval_test, Name=f'{model} & SMOTE method & {select_method}')


# feature selection 이후 파트 다시 정리하기


for select_method in feat_df.columns[1:]:

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
    for model_name, model in models:

        # 데이터 초기화
        y_data = df['관리종목여부']
        x_data = df.drop(["관리종목여부"], axis=1)

        x_data_col = return_selected_data(feat_df, model_name, select_method)  # 모델별 feature select 방식별 선정된 feature 반환
        # data augmentation 이전에 미리 원본 데이터로 테스트 데이터셋 구축(stratify 기능을 통해 균형 있게 테스트셋 구축)
        # ' 당기순이익(손실)' 처리
        if ' 당기순이익(손실)' in x_data_col:
            idx3 = x_data_col.index(' 당기순이익(손실)')
            x_data_col[idx3] = '당기순이익'

        if '유보액/납입자본비율' in x_data_col:
            idx4 = x_data_col.index('유보액/납입자본비율')
            x_data_col[idx4] = '유보액_납입자본비율'

        # feautre 선별
        new_x_data = x_data[x_data_col]
        new_y_data = y_data

        # 날짜 컬럼 빼주기
        if '날짜' in new_x_data.columns:
            new_x_data = new_x_data.drop('날짜', axis=1)

        # stratify를 통해 test 검증셋과 증강할 데이터셋 구분
        x_train, x_test, y_train, y_test = train_test_split(new_x_data, new_y_data, test_size=0.2, random_state=1123,
                                                            stratify=y_data)  # 데이터를 균등하게 나누고 test 검증하기 위해

        model_eval_train, model_eval_test, cf_matrix_train, cf_matrix_test = smote_eval(model, x_train, x_test, y_train,
                                                                                        y_test, select_method)


# ADASYN 적용
def adasyn_eval(model, x_train, x_test, y_train, y_test, select_method):
    from imblearn.over_sampling import ADASYN
    adasyn = ADASYN(random_state=1123)

    X_train_ada, y_train_ada = adasyn.fit_resample(x_train, y_train)
    print('ADASYN 적용 전 학습용 피처/레이블 데이터 세트:', x_train.shape, y_train.shape)
    print('ADASYN 적용 후 학습용 피처/레이블 데이터 세트:', X_train_ada.shape, y_train_ada.shape)
    print('ADASYN 적용 후 레이블 값 분포: \n', pd.Series(y_train_ada).value_counts())

    clf = build_model(model, X_train_ada, y_train_ada)  # 모델 학습

    # pred = model.predict(x_test)
    # pred_proba = model.predict_proba(x_test)
    # # cross_val_score(model, X_train_sm, y_train_sm, cv=skf, scoring='roc_auc').mean()

    # print('Confusion matrix')
    # print(confusion_matrix(y_test, pred))

    # print(f"Precision : {precision_score(y_test, pred, average='macro'):.4f}")
    # print(f"Recall    : {recall_score(y_test, pred, average='macro'):.4f}")
    # print(f"F1        : {f1_score(y_test, pred, average='macro'):.4f}")
    # print(f"Accuracy  : {accuracy_score(y_test, pred):.4f}")

    return model_eval_data(clf, X_train_ada, y_train_ada, x_test, y_test, model_eval_train,
                           model_eval_test, Name=f'{model} & ADASYN method & {select_method}')


# feature selection 이후 파트 다시 정리하기

for select_method in feat_df.columns[1:]:

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

    for model_name, model in models:

        # 데이터 초기화
        y_data = df['관리종목여부']
        x_data = df.drop(["관리종목여부"], axis=1)

        x_data_col = return_selected_data(feat_df, model_name, select_method)  # 모델별 feature select 방식별 선정된 feature 반환
        # data augmentation 이전에 미리 원본 데이터로 테스트 데이터셋 구축(stratify 기능을 통해 균형 있게 테스트셋 구축)
        # ' 당기순이익(손실)' 처리
        if ' 당기순이익(손실)' in x_data_col:
            idx3 = x_data_col.index(' 당기순이익(손실)')
        x_data_col[idx3] = '당기순이익'

        if '유보액/납입자본비율' in x_data_col:
            idx4 = x_data_col.index('유보액/납입자본비율')
        x_data_col[idx4] = '유보액_납입자본비율'

        # feautre 선별
        new_x_data = x_data[x_data_col]
        new_y_data = y_data

        # 날짜 컬럼 빼주기
        if '날짜' in new_x_data.columns:
            new_x_data = new_x_data.drop('날짜', axis=1)

            # stratify를 통해 test 검증셋과 증강할 데이터셋 구분
        x_train, x_test, y_train, y_test = train_test_split(new_x_data, new_y_data, test_size=0.2, random_state=1123,
                                                            stratify=y_data)  # 데이터를 균등하게 나누고 test 검증하기 위해

        model_eval_train, model_eval_test, cf_matrix_train, cf_matrix_test = adasyn_eval(model, x_train, x_test,
                                                                                         y_train, y_test, select_method)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Embedding
from keras.layers import LeakyReLU
# from keras.layers.advanced_activations import LeakyReLU
from keras.layers import ELU, PReLU, LeakyReLU
from tensorflow.keras.layers import concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
# from keras.layers.advanced_activations import LeakyReLU
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
import scipy.stats
import datetime as dt
import pydot
import warnings
warnings.filterwarnings("ignore")


encoder = LabelEncoder()

import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()



######################### GAN
def gan_before(selectioned_flist):
    # 데이터 준비
    df_minority_data = df.loc[df['관리종목여부'] == 1]  # 타겟==1인 애들만
    df_minority_data_withouttv = df_minority_data.loc[:, df_minority_data.columns != '관리종목여부']  # 타겟값을 제외한 전체

    # numberical_df, categorical_df 만들어주기
    numerical_list = []
    categorical_list = []
    globals()['numcat_list'] = []  # 나중에 쓰일 곳이 있다. -> 나중에 통합변수로 쓰나보다.

    for i in selectioned_flist:
        if df_minority_data_withouttv[i].dtypes == 'float64':  # numerical
            numerical_list.append(i)
            globals()['numcat_list'].append(i)  # 이 부분 살짝 수정
            globals()['numerical_df'] = df_minority_data_withouttv[numerical_list]  # 수치형만 여기에다

        else:
            categorical_list.append(i)
            globals()['numcat_list'].append(i)
            globals()['categorical_df'] = df_minority_data_withouttv[categorical_list]

    # 수치형 만들어주기
    globals()['generator_input_list'] = []
    for i in globals()['numerical_df'].columns:
        globals()[f'{i}_numerical'] = pd.DataFrame(numerical_df[i])  # 이 부분도 살짝 수정
        globals()['generator_input_list'].append(eval(f'{i}_numerical.shape[1]'))  # generator_input_list에서 1을 반환

    # 범주형 만들어주기
    for i in globals()['categorical_df'].columns:
        globals()[f'{i}_categorical'] = pd.DataFrame(categorical_df[i])  # shape를 찍기 위해??
        generator_input_list.append(eval(f'{i}_categorical.shape[1]'))

    ## GAN TRAIN INPUT LIST 만들기

    # gantrain_input_list 수치형 만들어주기
    globals()['gantrain_input_list'] = []
    for i in globals()['numerical_df'].columns:  # 수치형 df 컬럼 돌면서
        gantrain_input_list.append(eval(f'{i}_numerical.values'))

    # gantrain_input_list 범주형 만들어주기
    for i in globals()['categorical_df'].columns:
        gantrain_input_list.append(eval(f'{i}_categorical.values'))

    # 이건 뭘까
    globals()['numerical_catsh_list'] = []
    categorical_catsh_list = []  # 확인용
    globals()['all_catsh_list'] = []
    for i in range(1, len(globals()[
                              'numerical_df'].columns) + 1):  ################????????????? 이거 왜 1부터 돌아??????????????????????? numerical에서 n-1개 넣는다.
        globals()['numerical_catsh_list'].append(f'catsh{i}')  # catsh는 비어있는 그냥 변수
        all_catsh_list.append(f'catsh{i}')

    # categorical
    for i in range(len(globals()[f'numerical_catsh_list']) + 1, len(globals()['categorical_df'].columns) + len(
            globals()[f'numerical_catsh_list']) + 1):  # len(numerical_catsh_list)+1
        categorical_catsh_list.append(f'catsh{i}')
        all_catsh_list.append(f'catsh{i}')

    return generator_input_list


def gan_generator():
    # numerical hidden
    for i in range(1, len(globals()['numerical_df'].columns) + 1):  # numerical 안막을거면 +2 해줘야 함!
        if i == 1:
            noise = Input(shape=(3902,))  # 관리종목여부 ==0인 기업이 5000개이므로 4884개의 가상 데이터 생성 ㄱㄱ (1:1 비율로!) / try2: 3902
            globals()[f'hidden_{i}'] = Dense(8, kernel_initializer="he_uniform")(noise)
            globals()[f'hidden_{i}'] = LeakyReLU(0.2)(globals()[f'hidden_{i}'])
            globals()[f'hidden_{i}'] = BatchNormalization(momentum=0.8)(globals()[f'hidden_{i}'])
        else:
            globals()[f'hidden_{i}'] = Dense(8, kernel_initializer="he_uniform")(globals()[f'hidden_{i - 1}'])
            globals()[f'hidden_{i}'] = LeakyReLU(0.2)(globals()[f'hidden_{i}'])
            globals()[f'hidden_{i}'] = BatchNormalization(momentum=0.8)(globals()[f'hidden_{i}'])

    # categorical hidden
    for i in range(len(globals()['numerical_df'].columns) + 1,
                   len(globals()['categorical_df'].columns) + 1 + len(globals()['numerical_df'].columns)):
        globals()[f'hidden_{i}'] = Dense(8, kernel_initializer="he_uniform")(globals()[f'hidden_{i - 1}'])
        globals()[f'hidden_{i}'] = LeakyReLU(0.2)(globals()[f'hidden_{i}'])
        globals()[f'hidden_{i}'] = BatchNormalization(momentum=0.8)(globals()[f'hidden_{i}'])

    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # numerical branch
    for i in range(1, len(globals()['numerical_df'].columns) + 1):
        globals()[f'branch_{i}'] = Dense(32, kernel_initializer="he_uniform")(globals()[f'hidden_{i}'])
        globals()[f'branch_{i}'] = LeakyReLU(0.2)(globals()[f'branch_{i}'])
        globals()[f'branch_{i}'] = BatchNormalization(momentum=0.8)(globals()[f'branch_{i}'])
        globals()[f'branch_{i}'] = Dense(64, kernel_initializer="he_uniform")(globals()[f'branch_{i}'])
        globals()[f'branch_{i}'] = LeakyReLU(0.2)(globals()[f'branch_{i}'])
        globals()[f'branch_{i}'] = BatchNormalization(momentum=0.8)(globals()[f'branch_{i}'])

        # categorical branch
    for i in range(len(globals()['numerical_df'].columns) + 1,
                   len(globals()['categorical_df'].columns) + 1 + len(globals()['numerical_df'].columns)):
        globals()[f'branch_{i}'] = Dense(32, kernel_initializer="he_uniform")(globals()[f'hidden_{i}'])
        globals()[f'branch_{i}'] = LeakyReLU(0.2)(globals()[f'branch_{i}'])
        globals()[f'branch_{i}'] = BatchNormalization(momentum=0.8)(globals()[f'branch_{i}'])
        globals()[f'branch_{i}'] = Dense(64, kernel_initializer="he_uniform")(globals()[f'branch_{i}'])
        globals()[f'branch_{i}'] = LeakyReLU(0.2)(globals()[f'branch_{i}'])
        globals()[f'branch_{i}'] = BatchNormalization(momentum=0.8)(globals()[f'branch_{i}'])
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # numerical
    # for i in range(1,len(globals()['numerical_df'].columns)+1): # 손보영 이부분 틀림
    #   for a in globals()['numerical_df'].columns:
    #     globals()[f'catsh{i}'] = globals()[f'{a}_numerical'].shape[1]
    # # categorical
    # for i in range(len(globals()['numerical_df'].columns)+1,len(globals()['categorical_df'].columns)+1+len(globals()['numerical_df'].columns)):
    #   for a in globals()['categorical_df'].columns:
    #     globals()[f'catsh{i}'] = globals()[f'{a}_categorical'].shape[1]

    # numerical
    cnt = 1
    for a in globals()['numerical_df'].columns:
        globals()[f'catsh{cnt}'] = globals()[f'{a}_numerical'].shape[1]
        cnt += 1

    # categorical
    for a in globals()['categorical_df'].columns:
        globals()[f'catsh{cnt}'] = globals()[f'{a}_categorical'].shape[1]
        cnt += 1

    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # numerical
    all_branchoutput_list = []
    numerical_branchoutput_list = []  # 확인용
    for i in range(1, len(globals()['numerical_df'].columns) + 1):
        globals()[f'branch_{i}_output'] = Dense(globals()[f'catsh{i}'], activation="swish")(
            globals()[f'branch_{i}'])  # globals()[f'catsh{i}']  / globals()[f'branch_{i}'] # softmax
        numerical_branchoutput_list.append(f'branch_{i}_output')
        all_branchoutput_list.append(f'branch_{i}_output')
    # categorical
    categorical_branchoutput_list = []  # 확인용
    for i in range(len(globals()['numerical_df'].columns) + 1,
                   len(globals()['categorical_df'].columns) + 1 + len(globals()['numerical_df'].columns)):
        globals()[f'branch_{i}_output'] = Dense(globals()[f'catsh{i}'], activation="hard_sigmoid")(
            globals()[f'branch_{i}'])  # globals()[f'catsh{i}']  / globals()[f'branch_{i}'] # softmax //
        categorical_branchoutput_list.append(f'branch_{i}_output')
        all_branchoutput_list.append(f'branch_{i}_output')
    # return numerical_branchoutput_list , categorical_branchoutput_list , all_branchoutput_list
    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # combined_output = concatenate([branch_1_output, branch_2_output, branch_3_output,branch_4_output,branch_5_output,branch_6_output,branch_7_output,branch_8_output,branch_9_output,branch_10_output,branch_11_output,branch_12_output,branch_13_output])
    combined_output = branch_1_output
    for i in range(2, len(all_branchoutput_list) + 1):
        combined_output = concatenate([combined_output, globals()[f'branch_{i}_output']])  #
    return Model(inputs=noise, outputs=combined_output)


def gan_discriminator(inputs_n):
  # Input from generator
  d_input = Input(shape = (inputs_n,))
  d = Dense(128, kernel_initializer='he_uniform')(d_input)
  d = LeakyReLU(0.2)(d)
  d = Dense(64, kernel_initializer = 'he_uniform')(d)
  d = LeakyReLU(0.2)(d)
  d = Dense(32, kernel_initializer='he_uniform')(d)
  d = LeakyReLU(0.2)(d)
  d = Dense(16, kernel_initializer='he_uniform')(d)
  d = LeakyReLU(0.2)(d)
  d = Dense(8, kernel_initializer='he_uniform')(d)
  d = LeakyReLU(0.2)(d)

  # Output Layer
  d_output = Dense(1, activation = 'sigmoid')(d)

  # compile and return model
  model = Model(inputs = d_input, outputs = d_output)
  model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.0002, beta_1 = 0.5))
  return model

def define_complete_gan(generator, discriminator):

  gan_output = discriminator(generator.output)
  # Initialize gan
  model = Model(inputs = generator.input, outputs = gan_output)
  # Model Compilation
  model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr=0.0002, beta_1=0.5))
  discriminator.trainable = True # 학습 막고 싶으면 false
  return model


def gan_train(gan, generator, discriminator, latent_dim, n_epochs, n_batch, n_eval):
    half_batch = int(n_batch / 2)  # half_batch는 왜하는 걸까
    discriminator_loss = []
    generator_loss = []
    valid = np.ones((half_batch, 1))
    y_gan = np.ones((n_batch, 1))
    fake = np.zeros((half_batch, 1))
    # n_epochs = 10000 # 나중에 여러 에포크로 시도해보기
    for i in range(n_epochs):  # ----------------> 이부분 다시 봐보자
        # select random batch from real categorical and numerical data
        idx = np.random.randint(0, 116,
                                half_batch)  ###################################################################### 이부분 0 ~ 116까지 수 범위로 n_epochs//2 만큼 배열 채우기

        concatenate_list = []
        # categorical222_list = []
        # for num, j in enumerate(globals()['numcat_list']): # 모든 컬럼 이름을 모아놓은 것.
        #   if j in list(globals()['numerical_df'].columns): #numerical 변수면
        #     globals()[f'{j}_real'] = globals()[f'{j}_numerical'].values[idx]
        #     concatenate_list.append( globals()[f'{j}_real'])
        #   else: # 카테고리컬 변수면
        #     globals()[f'{j}_real'] = globals()[f'{j}_categorical'].values[idx]
        #     categorical222_list.append( globals()[f'{j}_real'])

        for j in globals()['numerical_df'].columns:
            globals()[f'{j}_real'] = globals()[f'{j}_numerical'].values[idx]
            concatenate_list.append(globals()[f'{j}_real'])

        for j in globals()['categorical_df'].columns:
            globals()[f'{j}_real'] = globals()[f'{j}_categorical'].values[idx]
            concatenate_list.append(globals()[f'{j}_real'])

            # concatenate_list.extend(categorical222_list)

        real_data = np.concatenate(concatenate_list, axis=1)

        # generate fake samples from the noise
        noise = np.random.normal(0, 1, (half_batch, latent_dim))  ############################ latent_dim은 어따가 쓰는겨??
        fake_data = generator.predict(noise)

        # train the discriminator and return losses and acc
        d_loss_real = da_real = discriminator.train_on_batch(real_data, valid)
        d_loss_fake = da_fake = discriminator.train_on_batch(fake_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        discriminator_loss.append(d_loss)

        # generate noise for generator input and train the generator (to have the discriminator label samples as valid)
        noise = np.random.normal(0, 1, (n_batch, latent_dim))
        g_loss = gan.train_on_batch(noise, y_gan)
        generator_loss.append(g_loss)

        # evaluate progress
        if (i + 1) % n_eval == 0:
            print("Epoch: %d [Discriminator loss: %f] [Generator loss: %f]" % (i + 1, d_loss, g_loss))

            # 시각화 하고싶으면 열어~~~~
    # plt.figure(figsize = (20, 10))
    # plt.plot(generator_loss, label = "Generator loss")
    # plt.plot(discriminator_loss, label = "Discriminator loss")
    # plt.title("Stats from training GAN")
    # plt.grid()
    # plt.legend()


# 아래 코드 자동화
def gan_afteone():
    noise = np.random.normal(0, 1, (3902, 3902))
    generated_mixed_data = generator.predict(noise)

    col_name = list(numerical_df.columns) + list(categorical_df.columns)

    globals()['mixed_gen_df'] = pd.DataFrame(data=generated_mixed_data, columns=col_name)

    ######################################## 시각화할거면 넣어~~ ####################################3
    # original_df = pd.DataFrame()
    # for i in globals()['numerical_df'].columns:
    #   original_df = pd.concat([original_df, globals()[f'{i}_numerical']], axis=1)
    # for i in globals()['categorical_df'].columns:
    #   original_df = pd.concat([original_df, globals()[f'{i}_categorical']], axis=1)

    # def normal_distribution(org, noise):
    #     org_x = np.linspace(org.min(), org.max(), len(org))
    #     noise_x = np.linspace(noise.min(), noise.max(), len(noise))
    #     org_y = scipy.stats.norm.pdf(org_x, org.mean(), org.std())
    #     noise_y = scipy.stats.norm.pdf(noise_x, noise.mean(), noise.std())
    #     n, bins, patches = plt.hist([org, noise], density = True, alpha = 0.5, color = ["green", "red"])
    #     xmin, xmax = plt.xlim()
    #     plt.plot(org_x, org_y, color = "green", label = "Original data", alpha = 0.5)
    #     plt.plot(noise_x, noise_y, color = "red", label = "Generated data", alpha = 0.5)
    #     title = f"Original data mean {np.round(org.mean(), 4)}, Original data std {np.round(org.std(), 4)}, Original data var {np.round(org.var(), 4)}\nGenerated data mean {np.round(noise.mean(), 4)}, Generated data {np.round(noise.std(), 4)}, Generated data var {np.round(noise.var(), 2)}"
    #     plt.title(title)
    #     plt.legend()
    #     plt.grid()
    #     plt.show()
    #     plt.close()

    # Numeric_columns=numerical_df.columns

    # for column in numerical_df.columns:
    #     print(column, "Comparison between Original Data and Generated Data")
    #     normal_distribution(original_df[column], mixed_gen_df[column])


def return_gan_data():  # gan_data 뱉어내기(학습용 데이터)
    def binary_step(data):
        # for i in range(len(data)):
        if data < 0.5:
            return 0
        else:
            return 1

    for i in globals()['categorical_df'].columns:
        globals()['mixed_gen_df'][f'{i}'] = globals()['mixed_gen_df'][f'{i}'].apply(binary_step)  # 0,1로 다 바꿔주기
        globals()['mixed_gen_df'][f'{i}'] = globals()['mixed_gen_df'][f'{i}'].astype('int64')  # type변환

    # df_generated_data 생성
    ## numerical_data >> generated_data에 넣어주기
    df_generated_data = pd.DataFrame()
    for i in globals()['numerical_df'].columns:
        df_generated_data[f'{i}'] = globals()['mixed_gen_df'][f'{i}']

    ## categorical_data >> generated_data에 넣어주기
    for i in globals()['categorical_df'].columns:
        df_generated_data[f'{i}'] = globals()['mixed_gen_df'][f'{i}']
        ## 타겟값 붙여주기
    df_generated_data['관리종목여부'] = 1
    # 이 다음이 원본데이터에서 모델학습에 쓸 피쳐들 따로 떼서 final_df 만들어주는거 >> 이건 자동화 밖에서 이미 설정해줄거라 포함안시켜도 될 듯

    # 생성된 데이터 X1, y1으로 나뉘어주고
    y1 = df_generated_data['관리종목여부']
    X1 = df_generated_data.drop(["관리종목여부"], axis=1)

    return X1, y1  # gan을 이용한 증강 데이터 반환(학습용 데이터로 쓸거임.)


# 데이터 초기화
df2 = pd.read_csv('./data/real_final_data_for_modeling.csv')
df2.rename(columns={' 당기순이익(손실)': '당기순이익'}, inplace=True)
df2.rename(columns={'유보액/납입자본비율': '유보액_납입자본비율'}, inplace=True)
# df.rename(columns={' 수익성':'수익성'}, inplace=True)


# 카테고리컬인 애들은 다 astype으로 미리 바꿔주자 >> 다 int64임
df2[['성장성', '안정성', '활동성', '수익성', '현금흐름', '부실기업판별_Z_score', 'F_score', '부실기업판별_K_score', '자본잠식', '벌금', '불성실공시', '소송',
     '영업정지', '특허',
     '투자주의환기종목', '업종', '관리종목요건_감사의견', '관리종목요건_매출액', '관리종목요건_영업손실', '관리종목요건_법인세비용차감전계속사업손실']] = df2[
    ['성장성', '안정성', '활동성', '수익성', '현금흐름', '부실기업판별_Z_score', 'F_score', '부실기업판별_K_score', '자본잠식', '벌금', '불성실공시', '소송',
     '영업정지', '특허',
     '투자주의환기종목', '업종', '관리종목요건_감사의견', '관리종목요건_매출액', '관리종목요건_영업손실', '관리종목요건_법인세비용차감전계속사업손실']].astype('int64')
df2['관리종목여부'] = df2['관리종목여부'].astype('int64')

for select_method in feat_df.columns[1:]:

    cat_clf = CatBoostClassifier(random_state=1123)
    lda_clf = LDA(n_components=1)
    RF_clf = RandomForestClassifier(random_state=1123)
    EX_clf = ExtraTreesClassifier(random_state=1123)
    LGB_clf = LGBMClassifier(random_state=1123)
    GB_clf = GradientBoostingClassifier(random_state=1123)

    models = [  # ("CF", cat_clf),
        # ("LDA", lda_clf),
        # ("RF", RF_clf),
        ("EX", EX_clf)
        # ("LGB", LGB_clf)
        # ('GB', GB_clf)
    ]

    for model_name, model in models:

        df = df2.copy()

        x_data_col = return_selected_data(feat_df, model_name, select_method)  # 모델별 feature select 방식별 선정된 feature 반환
        # data augmentation 이전에 미리 원본 데이터로 테스트 데이터셋 구축(stratify 기능을 통해 균형 있게 테스트셋 구축)
        # ' 당기순이익(손실)' 처리
        if ' 당기순이익(손실)' in x_data_col:
            idx3 = x_data_col.index(' 당기순이익(손실)')
            x_data_col[idx3] = '당기순이익'

        if '유보액/납입자본비율' in x_data_col:
            idx4 = x_data_col.index('유보액/납입자본비율')
            x_data_col[idx4] = '유보액_납입자본비율'

        # feautre 선별
        X = df[x_data_col]
        y = df['관리종목여부']

        # 날짜 컬럼 빼주기
        if '날짜' in list(X.columns):
            X = X.drop('날짜', axis=1)

        # 미리 테스트셋 만들기
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1123, stratify=y)

        # gan 관련 함수 실행
        gan_before(list(X_train.columns))
        generator = gan_generator()
        inputs_n = sum(globals()['generator_input_list'])
        discriminator = gan_discriminator(inputs_n)
        completegan = define_complete_gan(generator, discriminator)
        latent_dim = 3902
        gan_train(completegan, generator, discriminator, latent_dim, n_epochs=2000, n_batch=63,
                  n_eval=200)  ####################### 이거 나중에 바꿔줘
        gan_afteone()  # mixed 데이터 만들기

        # gan으로 학습데이터 만들기
        gan_X, gan_y = return_gan_data()  # 타겟값 1인 데이터 3902개씩 만들기

        ############################################################# concat 이전에 컬럼 형태 맞춰줘야한다!!!!!!!!!!!!!!!!!!!!!
        gan_X = gan_X[X_train.columns]

        # 기존 train 데이터에 합치기
        new_train_X = pd.concat([X_train, gan_X], axis=0)
        new_train_X = new_train_X.reset_index(drop=True)  # 인덱스 재배열

        new_train_y = np.concatenate((y_train, gan_y))

        # 학습
        clf = build_model(model, new_train_X, new_train_y)  # 모델 학습

        # 여기서 X_test 컬럼 순서를 new_train_X 컬럼으로 맞춰줘야한다.!!! 왜냐면 gan 돌릴때 순서가 numerical 다음 categorical 변수가 나오므로
        X_test = X_test[new_train_X.columns]

        pred = clf.predict(X_test)

        print(f'{model_name} & {select_method}의 성능평가점수')
        print(confusion_matrix(y_test, pred))
        print(f1_score(y_test, pred, average='macro'))

        model_eval_train, model_eval_test, cf_matrix_train, cf_matrix_test = model_eval_data(clf, new_train_X,
                                                                                             new_train_y, X_test,
                                                                                             y_test, model_eval_train,
                                                                                             model_eval_test,
                                                                                             Name=f'{model} & GAN method & {select_method}')


