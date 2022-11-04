'''
1) Feature selection

2)
'''
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, FRECV
from catboost import CatBoostClassifier
#from



#def feature_selection(df, method):

def rfe(model, x_data, y_data, ratio=0.9, min_feats=10):  # 어떤 모델을 넣느냐에 따라 다르다. 최소 컬럼 개수 10개로 설정.
    feats = x_data.columns.tolist()
    archive = pd.DataFrame(columns=['model', 'n_feats', 'feats', 'score'])
    while True:
        # model = CatBoostClassifier(random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_data[feats], y_data, random_state=42)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=100, verbose=False)
        val_pred = model.predict(x_val)

        # score = roc_auc_score(y_val, val_pred)
        score = f1_score(y_val, val_pred)  # f1 score를 기준으로
        n_feats = len(feats)
        print(n_feats, score)
        archive = archive.append({'model': model, 'n_feats': n_feats, 'feats': feats, 'score': score},
                                 ignore_index=True)
        feat_imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        next_n_feats = int(n_feats * ratio)
        if next_n_feats < min_feats:
            break
        else:
            feats = feat_imp.iloc[:next_n_feats].index.tolist()
    return archive
