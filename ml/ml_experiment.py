# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

import warnings
from ml.pre_handle import *
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
# 传统机器学习所使用的模型
class ML_model:
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    def svm(self, c=5, g = 3):
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        import scipy as sc
        #  索引最好的c = 5 gamma =1
        # bag of words c = le3 gamma = 0.1
        # tfidf c=2.8, g=2
        svc = make_pipeline(SVC(kernel='rbf', C=c, gamma=g))
        svc.fit(self.X_train,self.y_train)
        print("the model is svm and the valid's f1 is: ", f1_score(self.y_test, svc.predict(self.X_test),average="macro"))
        print("the model is svm and the valid's precision_score is: ", precision_score(self.y_test, svc.predict(self.X_test),average="macro"))
        print("the model is svm and the valid's recall_score is: ", recall_score(self.y_test, svc.predict(self.X_test),average="macro"))
        return svc

    def rf(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import RandomForestClassifier
        import scipy as sc
        rf = make_pipeline(RandomForestClassifier(random_state=590,n_estimators =6))
        rf.fit(self.X_train,self.y_train)
        print("the model is rf and the valid's f1 is: ", f1_score(self.y_test, rf.predict(self.X_test),average="macro"))
        print("the model is rf and the valid's precision_score is: ", precision_score(self.y_test, rf.predict(self.X_test),average="macro"))
        print("the model is rf and the valid's recall_score is: ", recall_score(self.y_test, rf.predict(self.X_test),average="macro"))
        return rf

    def gboost(self):
        from sklearn.ensemble import GradientBoostingClassifier
        import scipy as sc
        GBoost = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01,
                                           max_depth=12, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=97,
                                            random_state =200)
        GBoost.fit(self.X_train,self.y_train)
        print("the model is GBoost and the valid's f1 is: ", f1_score(self.y_test, GBoost.predict(self.X_test),average="macro"))
        print("the model is GBoost and the valid's precision_score is: ", precision_score(self.y_test, GBoost.predict(self.X_test),average="macro"))
        print("the model is GBoost and the valid's recall_score is: ", recall_score(self.y_test, GBoost.predict(self.X_test),average="macro"))
        return GBoost

    def xgboost(self):
        import xgboost as xgb
        import scipy as sc
        #     for i in range(0,1000,10):
        model_xgb = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=10,
                                     learning_rate=0.01, max_depth=11,
                                     min_child_weight=1.7817, n_estimators=500,
                                     reg_alpha=0.01, reg_lambda=5,
                                     subsample=0.5213, silent=1,
                                     seed =1024, nthread = -1)
        model_xgb.fit(self.X_train,self.y_train)
        print("the model is xgb and the valid's f1 is: ", f1_score(self.y_test, model_xgb.predict(self.X_test),average="macro"))
        print("the model is xgb and the valid's precision_score is: ", precision_score(self.y_test, model_xgb.predict(self.X_test),average="macro"))
        print("the model is xgb and the valid's recall_score is: ", recall_score(self.y_test, model_xgb.predict(self.X_test),average="macro"))
        return model_xgb
    def lgb(self):
        import lightgbm as lgb
        from lightgbm import LGBMClassifier
        import scipy as sc
        model_lgb = LGBMClassifier(num_leaves=5,
                                  learning_rate=0.05, n_estimators=550,
                                  max_bin = 25, bagging_fraction = 1,
                                  bagging_freq = 5, feature_fraction = 0.7,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf =42, min_sum_hessian_in_leaf = 40)
        model_lgb.fit(self.X_train,self.y_train)
        print("the model is model_lgb and the valid's f1 is: ", f1_score(self.y_test, model_lgb.predict(self.X_test),average="macro"))
        print("the model is model_lgb and the valid's precision_score is: ", precision_score(self.y_test, model_lgb.predict(self.X_test),average="macro"))
        print("the model is model_lgb and the valid's recall_score is: ", recall_score(self.y_test, model_lgb.predict(self.X_test),average="macro"))
        return model_lgb

    def stacking(self):
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler,MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
        from lightgbm import LGBMClassifier
        import xgboost as xgb
        from mlxtend.classifier import StackingClassifier
        import scipy as sc
        svc = make_pipeline(SVC(kernel='rbf', C=2.8, gamma=2))
        rf = RandomForestClassifier(random_state=590,n_estimators =6)
        GBoost = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01,
                                            max_depth=12, max_features='sqrt',
                                            min_samples_leaf=15, min_samples_split=97,
                                            random_state =200)
        model_xgb = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=10,
                                     learning_rate=0.01, max_depth=11,
                                     min_child_weight=1.7817, n_estimators=500,
                                     reg_alpha=0.01, reg_lambda=5,
                                     subsample=0.5213, silent=1,
                                     seed =1024, nthread = -1)
        model_lgb = LGBMClassifier(objective='regression',num_leaves=5,
                                   learning_rate=0.05, n_estimators=550,
                                   max_bin = 25, bagging_fraction = 1,
                                   bagging_freq = 5, feature_fraction = 0.7,
                                   feature_fraction_seed=9, bagging_seed=9,
                                   min_data_in_leaf =42, min_sum_hessian_in_leaf = 40)
        regressors = [rf,svc,GBoost, model_lgb,model_xgb]
        stregr = StackingClassifier(classifiers=regressors, meta_classifier=model_xgb,verbose=1)
        stregr.fit(self.X_train,self.y_train)
        print("the model is stregr and the valid's f1 is: ", f1_score(self.y_test, stregr.predict(self.X_test),average="macro"))
        # print("the model is stregr and the valid's precision_score is: ", precision_score(self.y_test, stregr.predict(self.X_test),average="macro"))
        # print("the model is stregr and the valid's recall_score is: ", recall_score(self.y_test, stregr.predict(self.X_test),average="macro"))
        return stregr

    class _AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

        def __init__(self, models):
            self.models = models
        # we define clones of the original models to fit the data in
        def fit(self, X, y):
            from sklearn.base import clone
            self.models_ = [clone(x) for x in self.models]
            # Train cloned base models
            for model in self.models_:
                model.fit(X, y)
            return self
        #Now we do the predictions for cloned models and average them
        def predict(self, X):
            predictions = np.column_stack([
                model.predict_prob(X) for model in self.models_
            ])
            return np.mean(predictions, axis=1)
    def average(self):
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler,MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
        from lightgbm import LGBMClassifier
        import xgboost as xgb
        from mlxtend.classifier import StackingClassifier
        from sklearn.kernel_ridge import KernelRidge
        import scipy as sc
        #         self._load_package()
        # c=7,g=0.075
        lasso = make_pipeline(SVC(kernel='rbf', C=2.8, gamma=2))
        rf = RandomForestClassifier(random_state=590,n_estimators =6)
        GBoost = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01,
                                           max_depth=12, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=97,
                                           random_state =200)
        model_xgb = xgb.XGBClassifier(colsample_bytree=0.4603, gamma=10,
                                     learning_rate=0.01, max_depth=11,
                                     min_child_weight=1.7817, n_estimators=500,
                                     reg_alpha=0.01, reg_lambda=5,
                                     subsample=0.5213, silent=1,
                                     seed =1024, nthread = -1)
        model_lgb = LGBMClassifier(num_leaves=5,
                                  learning_rate=0.05, n_estimators=550,
                                  max_bin = 25, bagging_fraction = 1,
                                  bagging_freq = 5, feature_fraction = 0.7,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf =42, min_sum_hessian_in_leaf = 40)
        regressors = [rf, lasso,GBoost, model_lgb, model_xgb]
        stregr = StackingClassifier(classifiers=regressors, meta_classifier=model_xgb)
        averaged_models =self._AveragingModels(models = (rf,stregr,model_xgb,lasso))
        averaged_models.fit(self.X_train,self.y_train)
        print("the model is averaged_models and the valid's f1 is: ", f1_score(self.y_test, averaged_models.predict(self.X_test),average="macro"))
        print("the model is averaged_models and the valid's precision_score is: ", precision_score(self.y_test, averaged_models.predict(self.X_test),average="macro"))
        print("the model is averaged_models and the valid's recall_score is: ", recall_score(self.y_test, averaged_models.predict(self.X_test),average="macro"))
        return averaged_models


def train():
    train, train_label, valid, valid_label, test, test_label = get_allData()
    model = ML_model(train, valid, train_label, valid_label)
    import warnings
    warnings.filterwarnings("ignore")
    # rf = model.rf()
    # print("the model is rf and the test's f1 is: ", f1_score(test_label, rf.predict(test),average="macro"))
    # print("the model is rf and the test's precision_score is: ", precision_score(test_label, rf.predict(test),average="macro"))
    # print("the model is rf and the test's recall_score is: ", recall_score(test_label, rf.predict(test),average="macro"))
    # print("----------------------------------------------------------------------------------------")
    gboost = model.gboost()
    print("the model is gboost and the test's f1 is: ", f1_score(test_label, gboost.predict(test),average="macro"))
    print("the model is gboost and the test's precision_score is: ", precision_score(test_label, gboost.predict(test),average="macro"))
    print("the model is gboost and the test's recall_score is: ", recall_score(test_label, gboost.predict(test),average="macro"))
    # print("----------------------------------------------------------------------------------------")
    # svm = model.svm()
    # print("the model is svm and the test's f1 is: ", f1_score(test_label, svm.predict(test),average="macro"))
    # print("the model is svm and the test's precision_score is: ", precision_score(test_label, svm.predict(test),average="macro"))
    # print("the model is svm and the test's recall_score is: ", recall_score(test_label, svm.predict(test),average="macro"))
    # print("----------------------------------------------------------------------------------------")
    # xbg = model.xgboost()
    # print("the model is xbg and the test's f1 is: ", f1_score(test_label, xbg.predict(test),average="macro"))
    # print("the model is xbg and the test's precision_score is: ", precision_score(test_label, xbg.predict(test),average="macro"))
    # print("the model is xbg and the test's recall_score is: ", recall_score(test_label, xbg.predict(test),average="macro"))
    # print("----------------------------------------------------------------------------------------")
    # lgb = model.lgb()
    # print("the model is lgb and the test's f1 is: ", f1_score(test_label, lgb.predict(test),average="macro"))
    # print("the model is lgb and the test's precision_score is: ", precision_score(test_label, lgb.predict(test),average="macro"))
    # print("the model is lgb and the test's recall_score is: ", recall_score(test_label, lgb.predict(test),average="macro"))
    # print("----------------------------------------------------------------------------------------")
    # stack = model.stacking()
    # print("the model is stack and the test's f1 is: ", f1_score(test_label, stack.predict(test),average="macro"))
    # print("the model is stack and the test's precision_score is: ", precision_score(test_label, stack.predict(test),average="macro"))
    # print("the model is stack and the test's recall_score is: ", recall_score(test_label, stack.predict(test),average="macro"))
    # print("----------------------------------------------------------------------------------------")
    average = model.average()
    print("the model is average and the test's f1 is: ", f1_score(test_label, average.predict(test),average="macro"))
    print("the model is average and the test's precision_score is: ", precision_score(test_label, average.predict(test),average="macro"))
    print("the model is average and the test's recall_score is: ", recall_score(test_label, average.predict(test),average="macro"))

if __name__ == '__main__':
    train()