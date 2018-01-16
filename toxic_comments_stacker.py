import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
import keras.backend as K
from keras.models import Model
from keras.layers import Dropout, GlobalMaxPool1D, SpatialDropout1D, multiply
from keras.layers import Lambda, Permute, Reshape, RepeatVector
from keras.layers import Bidirectional, CuDNNGRU, Dense, Embedding, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import get_glove_embeddings, load_and_tokenize

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
from xgboost import XGBClassifier, DMatrix


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # this line is not useful. It's just to know which dimension is what.
    a = Reshape((input_dim, MAX_LENGTH))(a)
    a = Dense(MAX_LENGTH, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')

    return output_attention_mul


def build_model(embedding_matrix, word_index):
    embedding_dim = embedding_matrix.shape[1]
    hidden_units = 128
    dense_units_1 = 256
    dense_units_2 = 128

    input_placeholder = Input(shape=(MAX_LENGTH, ))
    x = Embedding(len(word_index) + 1,
                  embedding_dim,
                  weights=[embedding_matrix],
                  input_length=MAX_LENGTH,
                  trainable=False)(input_placeholder)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(units=hidden_units,
                               return_sequences=True))(x)

    attention = attention_3d_block(x)

    dense = GlobalMaxPool1D()(attention)
    dense = Dropout(0.2)(dense)
    dense = Dense(dense_units_1, activation='elu')(dense)
    dense = Dropout(0.15)(dense)
    dense = Dense(dense_units_2, activation='elu')(dense)
    dense = Dropout(0.1)(dense)
    probs = Dense(6, activation='sigmoid')(dense)

    model = Model(input=input_placeholder, output=probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def fit_and_predict_oof(X_train, y_train, X_test,
                        base_model, n_splits=5):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    folds = list(KFold(n_splits=n_splits, shuffle=True).split(X_train, y_train))
    n_models = 1  # y_train.shape[1]
    n_classes = y_train.shape[1]

    S_train = np.zeros((X_train.shape[0], n_models*n_classes))
    S_test = np.zeros((X_test.shape[0], n_models*n_classes))

    for i, clf in enumerate([n_models]):
        print('\nFitting model {}/{}'.format(i + 1, n_models))
        model_start = time.time()
        S_test_i = np.zeros((X_test.shape[0], n_splits, n_classes))

        for j, (train_idx, val_idx) in enumerate(folds):
            print('\nFitting to fold {}/{}'.format(j + 1, n_splits))
            fold_start = time.time()

            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]

            fold_model = None
            fold_model = Model.from_config(base_model.get_config())
            fold_model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])
            # Redefine callbacks to get early stopping
            CP_callback = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1,
                                          save_best_only=True, save_weights_only=True, mode='min')
            ES_callback = EarlyStopping(monitor='val_loss', mode='min', patience=5)
            callbacks = [CP_callback, ES_callback]

            fold_model.fit(X_fold_train, y_fold_train,
                           batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, verbose=1,
                           validation_data=(X_val, y_val),
                           callbacks=callbacks)

            fold_model.load_weights(best_weights_path)

            y_pred = fold_model.predict(X_val)

            # .reshape((y_pred.shape[0], n_classes))
            S_train[val_idx, i:(i + 1) * n_classes] = y_pred
            # .reshape((X_test.shape[0], n_classes))
            S_test_i[:, j, :] = fold_model.predict(X_test)
            m, s = divmod(time.time() - fold_start, 60)
            print('Time to fit model to fold: {}m {}s'.format(int(m), int(s)))

        S_test[:, i:(i + 1)*n_classes] = S_test_i.mean(axis=1)
        np.squeeze(S_test)

        m, s = divmod(time.time() - model_start, 60)
        print('Time to fit model {} to all folds: {}m {}s'.format(i + 1, int(m), int(s)))

    print('Fitting stacker to train data and evaluating on test data')
    res = np.zeros((S_test.shape[0], n_classes))
    xgb_params = {
        'max_depth': 4,
        'learning_rate': 0.01,
        'n_estimators': 300,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'nthread': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
        }
    for c, col in enumerate(class_list):
        print('Fitting to category ' + str(col))
        stacker = XGBClassifier(**xgb_params)
        stacker.fit(S_train, y_train[:, c])
        res[:, c] = stacker.predict_proba(S_test)[:, 1]

    return res


MAX_NUM_WORDS = 30000
MAX_LENGTH = 200
EMBEDDING_DIM = 300

BATCH_SIZE = 256
MAX_EPOCHS = 25

SINGLE_ATTENTION_VECTOR = False
TRAIN = True
MAKE_SUBMISSION = True

class_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X_train, train_targets, X_test, word_index = load_and_tokenize(class_list,
                                                               MAX_NUM_WORDS,
                                                               MAX_LENGTH)

embedding_matrix = get_glove_embeddings(word_index, EMBEDDING_DIM, False)
model = build_model(embedding_matrix, word_index)
model.summary()

best_weights_path = 'weights_base.best.hdf5'

if TRAIN:
    print('\nTraining model and stacker...')
    y_test = fit_and_predict_oof(X_train, train_targets, X_test,
                                 model, n_splits=5)
    print('Training done')

if MAKE_SUBMISSION:
    sample_submission = pd.read_csv('./data/sample_submission.csv')
    sample_submission[class_list] = y_test
    sample_submission.to_csv('submissions/stacker_output.csv', index=False)
