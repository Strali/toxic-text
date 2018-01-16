import re

import pandas as pd
from keras.preprocessing import text, sequence

from feature_engineering import Tfidfize
from utils import get_sentence


def load_and_tokenize(class_list, max_nb_words, max_padding_length, track_index=0,
                      use_extra_features=False, use_extended_data=False):
    """
    Return training and test data preprocessed for toxicity classification.

    Parameters:
    -----------
    class_list : 1-D array of all classes in output.
    max_nb_words : Maximum number of words from corpus to use in tokenizer.
    max_padding_length : Maximum length of sentence representation; shorter
        sentences are padded and longer are truncated to this length.
    track_index : Index of a sentence for which to track attention activations.
    use_extended_data : Boolean, whether or not to use additional figshare
        toxic comment data in training data.

    Returns:
    --------
    X_train : Array of tokenized sentences for training.
    y_train : Array of training labels.
    X_test : Array of tokenized sentences for testing.
    word_index : word_index : List of all tokens (words) in the corpus.
    tracked_sentence : Raw text sentence for which to track attention.
    tracked_target : Labels for the tracked sentence.

    """

    print('\nLoading and tokenizing data...')
    if use_extra_features:
        train = pd.read_csv('./data/train_with_convai.csv')
        test = pd.read_csv('./data/test_with_convai.csv')
    else:
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')

    text_normalization(train)
    text_normalization(test)

    sample_text, sample_target = get_sentence(track_index,
                                              class_list,
                                              train)

    train_comments = train['comment_text'].values
    y_train = train[class_list].values
    test_comments = test['comment_text'].values

    if use_extra_features:
        train_aux = train[['toxic_level', 'attack', 'aggression']].values
        test_aux = test[['toxic_level', 'attack', 'aggression']].values

    train_tokenized, tfidfer_train = Tfidfize(train)
    test_tokenized, tfidfer_test = Tfidfize(test)
    print(test_tokenized[0])

    train_comments_tokenized = tokenizer.texts_to_sequences(train_comments)
    test_comments_tokenized = tokenizer.texts_to_sequences(test_comments)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X_train = sequence.pad_sequences(train_comments_tokenized, max_padding_length)
    X_test = sequence.pad_sequences(test_comments_tokenized, max_padding_length)

    print('Loaded data\n')

    if use_extra_features:
        return X_train, train_aux, y_train, X_test, test_aux, \
            word_index, sample_text, sample_target
    else:
        return X_train, y_train, X_test, word_index, sample_text, sample_target


def text_normalization(text_data):
    """
    Filter non-standard symbols and ip-adressesfrom text data,
        lowercase and strip whitespace.

    Parameters:
    -----------
    text_data : Pandas dataframe of text strings. It is assumed that this
        dataframe has a field called 'comment_text'
    """
    regex = r"[^a-zA-Z]"
    # ip_regex = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    # char_regex = r"[~â!@#$%^&*`·¢£¡¿ƒ€œ§µ¤“()_|¦©+\-=?—•;:\"\',.<>\{\}\[\]\\\/]"

    text_data['comment_text'] = text_data['comment_text'].fillna('MISSING')
    '''
    Processing done:
        - Remove special characters using regex
        - Remove excess whitespace
        - Remove URL https
        - Lowercase
    '''
    text_data['comment_text'] = text_data['comment_text'].apply(
        lambda x: re.sub(regex, ' ', x))
    text_data['comment_text'] = text_data['comment_text'].apply(
        lambda x: re.sub('\s+', ' ', x).strip())
    text_data['comment_text'] = text_data['comment_text'].apply(
        lambda x: x.replace("http", ""))
    text_data['comment_text'] = text_data['comment_text'].apply(
        lambda x: x.lower())
