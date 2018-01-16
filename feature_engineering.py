import gc
import re

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb


def JoinAndSanitize(cmt, annot):
    df = cmt.set_index('rev_id').join(annot.groupby(['rev_id']).mean())
    df = Sanitize(df)
    return df


def Sanitize(df):
    comment = 'comment' if 'comment' in df else 'comment_text'

    regex = r"[^a-zA-Z]"
    df[comment] = df[comment].fillna('unk')
    df[comment] = df[comment].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    df[comment] = df[comment].apply(lambda x: x.replace("TAB_TOKEN", " "))
    df[comment] = df[comment].apply(lambda x: re.sub(regex, ' ', x))
    df[comment] = df[comment].apply(lambda x: x.lower())
    df[comment] = df[comment].apply(lambda x: re.sub('\s+', ' ', x).strip())

    return df


def Tfidfize(df):
    max_vocab = 60000
    comment = 'comment' if 'comment' in df else 'comment_text'

    tfidfer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_vocab,
                              use_idf=1, stop_words='english',
                              smooth_idf=1, sublinear_tf=1)
    tfidf = tfidfer.fit_transform(df[comment])

    return tfidf, tfidfer


def TfidfAndPredict(tfidfer, model, train, test, USE_RIDGE=True, USE_RF=False):
    if USE_RIDGE or USE_RF:
        tfidf_train = tfidfer.transform(train['comment_text'])
        tfidf_test = tfidfer.transform(test['comment_text'])
        train_scores = model.predict(tfidf_train)
        test_scores = model.predict(tfidf_test)
    else:
        tfidf_train = tfidfer.transform(train['comment_text'])
        tfidf_test = tfidfer.transform(test['comment_text'])
        train_scores = model.predict(xgb.DMatrix(tfidf_train))
        test_scores = model.predict(xgb.DMatrix(tfidf_test))
        gc.collect()

    return train_scores, test_scores


def main():
    USE_RIDGE = True
    USE_RF = False
    USE_VAL_DATA = True

    toxic_comments = pd.read_csv('./data/toxicity_annotated_comments.tsv', sep='\t')
    toxic_annot = pd.read_csv('./data/toxicity_annotations.tsv', sep='\t')

    agg_comments = pd.read_csv('./data/aggression_annotated_comments.tsv', sep='\t')
    agg_annot = pd.read_csv('./data/aggression_annotations.tsv', sep='\t')

    attack_comments = pd.read_csv('./data/attack_annotated_comments.tsv', sep='\t')
    attack_annot = pd.read_csv('./data/attack_annotations.tsv', sep='\t')

    toxic = JoinAndSanitize(toxic_comments, toxic_annot)
    attack = JoinAndSanitize(attack_comments, attack_annot)
    aggression = JoinAndSanitize(agg_comments, agg_annot)

    X_toxic, tfidfer_toxic = Tfidfize(toxic)
    y_toxic = toxic['toxicity'].values
    X_attack, tfidfer_attack = Tfidfize(attack)
    y_attack = attack['attack'].values
    X_aggression, tfidfer_aggression = Tfidfize(aggression)
    y_aggression = aggression['aggression'].values

    print(y_toxic.shape)
    print(X_toxic.shape)
    if USE_RIDGE:
        print('Using ridge regression model')
        ridge = Ridge()

        mse_toxic = -cross_val_score(ridge, X_toxic, y_toxic, scoring='neg_mean_squared_error')
        mse_attack = -cross_val_score(ridge, X_attack, y_attack, scoring='neg_mean_squared_error')
        mse_aggression = -cross_val_score(ridge, X_aggression, y_aggression, 
                                          scoring='neg_mean_squared_error')
        print('MSE toxic: ', mse_toxic, 'MSE attack: ', mse_attack, 
              'MSE aggression: ', mse_aggression)

        print('Fitting ridge regression model to toxic labels')
        model_toxic = ridge.fit(X_toxic, y_toxic)
        print('Fitting ridge regression model to attack labels')
        model_attack = ridge.fit(X_attack, y_attack)
        print('Fitting ridge regression model to aggression labels')
        model_aggression = ridge.fit(X_aggression, y_aggression)

    elif USE_RF:
        print('Using random forest regression model')
        rf = RandomForestRegressor(n_estimators=100,
                                   max_depth=4,
                                   min_samples_leaf=25,
                                   oob_score=True)

        print('Fitting ridge regression model to toxic labels')
        model_toxic = rf.fit(X_toxic, y_toxic)
        print('Fitting ridge regression model to attack labels')
        model_attack = rf.fit(X_attack, y_attack)
        print('Fitting ridge regression model to aggression labels')
        model_aggression = rf.fit(X_aggression, y_aggression)

    else:
        print('Using XGBoost')
        xgb_params = {
            'max_depth': 4,
            'learning_rate': 0.01,
            'n_estimators': 400,
            'subsample': 0.7,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'nthread': -1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        num_rounds = 500

        X_toxic_train, X_toxic_test, y_toxic_train, y_toxic_test = \
            train_test_split(X_toxic, y_toxic, test_size=0.2)
        X_attack_train, X_attack_test, y_attack_train, y_attack_test = \
            train_test_split(X_attack, y_attack, test_size=0.2)
        X_aggro_train, X_aggro_test, y_aggro_train, y_aggro_test = \
            train_test_split(X_aggression, y_aggression, test_size=0.2)

        xg_toxic_train = xgb.DMatrix(X_toxic_train, y_toxic_train)
        xg_toxic_test = xgb.DMatrix(X_toxic_test, y_toxic_test)
        toxic_watchlist = [(xg_toxic_train, 'train'), (xg_toxic_test, 'test')]

        xg_attack_train = xgb.DMatrix(X_attack_train, y_attack_train)
        xg_attack_test = xgb.DMatrix(X_attack_test, y_attack_test)
        attack_watchlist = [(xg_attack_train, 'train'), (xg_attack_test, 'test')]

        xg_aggression_train = xgb.DMatrix(X_aggro_train, y_aggro_train)
        xg_aggression_test = xgb.DMatrix(X_aggro_test, y_aggro_test)
        aggro_watchlist = [(xg_aggression_train, 'train'), (xg_aggression_test, 'test')]

        if USE_VAL_DATA:
            model_toxic = xgb.train(params=xgb_params,
                                    dtrain=xg_toxic_train,
                                    evals=toxic_watchlist,
                                    num_boost_round=num_rounds,
                                    early_stopping_rounds=20)
            model_attack = xgb.train(params=xgb_params,
                                     dtrain=xg_attack_train,
                                     evals=attack_watchlist,
                                     num_boost_round=num_rounds,
                                     early_stopping_rounds=20)
            model_aggression = xgb.train(params=xgb_params,
                                         dtrain=xg_aggression_train,
                                         evals=aggro_watchlist,
                                         num_boost_round=num_rounds,
                                         early_stopping_rounds=20)

    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    train = Sanitize(train)
    test = Sanitize(test)

    print('Using model to predict on real data')

    if USE_RIDGE:
        toxic_tr_scores, toxic_t_scores = TfidfAndPredict(tfidfer_toxic, 
                                                          model_toxic,
                                                          train, test,
                                                          USE_RIDGE,
                                                          USE_RF)
        attack_tr_scores, attack_t_scores = TfidfAndPredict(tfidfer_attack,
                                                            model_attack,
                                                            train, test,
                                                            USE_RIDGE,
                                                            USE_RF)
        aggression_tr_scores, aggression_t_scores = TfidfAndPredict(
            tfidfer_aggression,
            model_aggression,
            train, test,
            USE_RIDGE,
            USE_RF)

        train['toxic_level'] = toxic_tr_scores
        train['attack'] = attack_tr_scores
        train['aggression'] = aggression_tr_scores
        test['toxic_level'] = toxic_t_scores
        test['attack'] = attack_t_scores
        test['aggression'] = aggression_t_scores

    else:
        toxic_tr_scores, toxic_t_scores = TfidfAndPredict(tfidfer_toxic,
                                                          model_toxic,
                                                          train, test,
                                                          USE_RIDGE,
                                                          USE_RF)
        attack_tr_scores, attack_t_scores = TfidfAndPredict(tfidfer_attack,
                                                            model_attack,
                                                            train, test,
                                                            USE_RIDGE,
                                                            USE_RF)
        aggression_tr_scores, aggression_t_scores = TfidfAndPredict(
            tfidfer_aggression,
            model_aggression,
            train, test,
            USE_RIDGE,
            USE_RF)

        train['toxic_level'] = toxic_tr_scores
        train['attack'] = attack_tr_scores
        train['aggression'] = aggression_tr_scores
        test['toxic_level'] = toxic_t_scores
        test['attack'] = attack_t_scores
        test['aggression'] = aggression_t_scores

    print('Done! Saving to file...')
    if USE_RIDGE:
        train.to_csv('./data/train_with_convai_ridge.csv', index=False)
        test.to_csv('./data/test_with_convai_ridge.csv', index=False)
    elif USE_RF:
        train.to_csv('./data/train_with_convai_rf.csv', index=False)
        test.to_csv('./data/test_with_convai_rf.csv', index=False)
    else:
        train.to_csv('./data/train_with_convai_xgb.csv', index=False)
        test.to_csv('./data/test_with_convai_xgb.csv', index=False)


if __name__ == '__main__':
    main()
