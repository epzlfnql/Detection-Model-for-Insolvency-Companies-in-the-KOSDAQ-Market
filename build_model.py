'''
Catboost - 장점
Catboost는 시계열 데이터를 효율적으로 처리하는 것으로 알려져있다.
또 속도가 매우 빨라 실제 상용화되고 있는 서비스에 예측 기능을 삽입하고자 할 때 효과적일 수 있다. XGBoost보다 예측 속도가 8배 빠르다고 알려져있다.
imbalanced dataset도 class_weight 파라미터 조정을 통해 예측력을 높일 수도 있다.
그 외 Catboost는 오버피팅을 피하기 위해 내부적으로 여러 방법(random permutation, overfitting detector)을 갖추고 있어 속도 뿐만 아니라 예측력도 굉장히 높다.
'''
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedKFold

import sys, os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
# Use below for charts in dark jupyter theme
THEME_DARK = False
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier,RandomForestRegressor
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings


