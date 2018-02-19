"""
TODO: Add docstrings
"""
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

    regex = r"[^a-zA-Z!]"
    u_regex = r'(?<![a-z])u(?![a-z])'
    http_regex = r'^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/).*[\r\n]*'
    email_regex = r'\S*@\S*\s?'

    df[comment] = df[comment].fillna('unk')
    df[comment] = df[comment].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    df[comment] = df[comment].apply(lambda x: x.replace("TAB_TOKEN", " "))
    df[comment] = df[comment].apply(lambda x: re.sub(http_regex, ' ', x))
    df[comment] = df[comment].apply(lambda x: re.sub(email_regex, ' ', x))
    df[comment] = df[comment].apply(lambda x: re.sub(u_regex, 'you', x, flags=re.IGNORECASE))
    df[comment] = df[comment].apply(lambda x: re.sub(regex, ' ', x))
    df[comment] = df[comment].apply(lambda x: re.sub('\!+', '! ', x))
    df[comment] = df[comment].apply(lambda x: re.sub('\s+', ' ', x).strip())
    df[comment] = df[comment].apply(lambda x: x.lower())

    return df


def Tfidfize(df):
    max_vocab = 60000
    comment = 'comment' if 'comment' in df else 'comment_text'

    tfidfer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_vocab,
                              use_idf=1, stop_words='english',
                              smooth_idf=1, sublinear_tf=1)
    tfidf = tfidfer.fit_transform(df[comment])

    return tfidf, tfidfer


def TfidfAndPredict(tfidfer, model, train, test):
    tfidf_train = tfidfer.transform(train['comment_text'])
    tfidf_test = tfidfer.transform(test['comment_text'])
    train_scores = model.predict(tfidf_train)
    test_scores = model.predict(tfidf_test)

    return train_scores, test_scores


def main():
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

    train = pd.read_csv('./data/train_de.csv')
    test = pd.read_csv('./data/test.csv')
    train = Sanitize(train)
    test = Sanitize(test)

    print('Using model to predict on real data')

    toxic_tr_scores, toxic_t_scores = TfidfAndPredict(tfidfer_toxic,
                                                        model_toxic,
                                                        train, test)
    attack_tr_scores, attack_t_scores = TfidfAndPredict(tfidfer_attack,
                                                        model_attack,
                                                        train, test)
    aggression_tr_scores, aggression_t_scores = TfidfAndPredict(
        tfidfer_aggression,
        model_aggression,
        train, test)

    train['toxic_level'] = toxic_tr_scores
    train['attack'] = attack_tr_scores
    train['aggression'] = aggression_tr_scores
    test['toxic_level'] = toxic_t_scores
    test['attack'] = attack_t_scores
    test['aggression'] = aggression_t_scores

    print('Done! Saving to file...')
    train.to_csv('./data/train_de_with_convai_ridge.csv', index=False)
    #test.to_csv('./data/test_with_convai_ridge.csv', index=False)


if __name__ == '__main__':
    main()
