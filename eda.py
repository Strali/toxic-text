import re
from statistics import mean, median

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nltk.tokenize import WhitespaceTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def standardize_text(df, text_field):
    regex = r'[^a-zA-Z!]'
    u_regex = r'(?<![a-z])u(?![a-z])'
    http_regex = r'^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/).*[\r\n]*'
    email_regex = r'\S*@\S*\s?'

    '''
    Processing done:
        - Remove URL strings
        - Remove email addresses
        - Remove special characters using regex, barring !-tokens
        - Replace multiple !:s with only one
        - Remove excess whitespace
        - Lowercase
    '''
    df[text_field] = df[text_field].apply(lambda x: re.sub(http_regex, ' ', x))
    df[text_field] = df[text_field].apply(lambda x: re.sub(email_regex, ' ', x))
    df[text_field] = df[text_field].apply(
        lambda x: re.sub(u_regex, 'you', x, flags=re.IGNORECASE))
    df[text_field] = df[text_field].apply(lambda x: re.sub(regex, ' ', x))
    df[text_field] = df[text_field].apply(lambda x: re.sub('\!+', '! ', x))
    df[text_field] = df[text_field].apply(lambda x: re.sub('\s+', ' ', x).strip())
    df[text_field] = df[text_field].apply(lambda x: x.lower())

    return df


def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    # loop for each class
    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i])
                            for i, el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key=lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }
    return classes


def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a, b) for a, b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])

    bottom_pairs = [(a, b) for a, b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)

    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]

    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]

    fig = plt.figure(figsize=(10, 10))

    plt.subplot(121)
    plt.barh(y_pos, bottom_scores, align='center', alpha=0.5)
    plt.title('Clean', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplot(122)
    plt.barh(y_pos, top_scores, align='center', alpha=0.5)
    plt.title('Toxic', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplots_adjust(wspace=0.8)
    plt.show()


def plot_length_histogram(sentence_lengths):
    fig = plt.figure(figsize=(10, 10))
    plt.title('Distribution of 95:th percentile sentence lengths')
    plt.xlabel('Sentence length')
    plt.ylabel('Number of sentences')
    plt.hist(sentence_lengths)
    plt.show()


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['green', 'orangered']
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8,
                    c=test_labels, cmap=mcolors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='orangered', label='Toxic')
        green_patch = mpatches.Patch(color='green', label='Clean')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer


CLASS_LIST = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
data = pd.read_csv('./data/train.csv')
data = standardize_text(data, 'comment_text')

print(data[CLASS_LIST].sum(axis=0))

tokenizer = WhitespaceTokenizer()
# tokenizer = RegexpTokenizer(r'\w+')

data['tokens'] = data['comment_text'].apply(tokenizer.tokenize)
print(data.head())

all_words = [word for tokens in data['tokens'] for word in tokens]
sentence_lengths = [len(tokens) for tokens in data['tokens']]
length_percentiles = np.percentile(sentence_lengths, [68, 95, 99])
VOCAB = sorted(list(set(all_words)))

print('%s words total, with a vocabulary size of %s' % (len(all_words), len(VOCAB)))
print('Average sentence length is %s' % mean(sentence_lengths))
print('Median sentence length is %s' % median(sentence_lengths))
print('Max sentence length is %s' % max(sentence_lengths))
print('68:th, 95:th and 99:th percentiles of sentence lengths are %s' % length_percentiles)
'''
plot_length_histogram([s for s in sentence_lengths if s <= length_percentiles[1]])
'''
text_list = data['comment_text'][0:50000].tolist()
data['is_toxic'] = data[CLASS_LIST].sum(axis=1)
data.loc[data['is_toxic'] > 1] = 1
label_list = data['is_toxic'][0:50000].tolist()

text_tfidf, tfidf_vectorizer = tfidf(text_list)
'''
fig = plt.figure(figsize=(12, 12))
plot_LSA(text_tfidf, label_list)
plt.show()
'''
clf = LogisticRegression(C=10.0, class_weight='balanced', solver='newton-cg',
                         multi_class='multinomial', n_jobs=1, random_state=42)
clf.fit(text_tfidf, label_list)

importance = get_most_important_features(tfidf_vectorizer, clf, 10)

top_scores = [a[0] for a in importance[0]['tops']]
top_words = [a[1] for a in importance[0]['tops']]
bottom_scores = [a[0] for a in importance[0]['bottom']]
bottom_words = [a[1] for a in importance[0]['bottom']]
plot_important_words(top_scores, top_words, bottom_scores,
                     bottom_words, "Most important words for relevance")
