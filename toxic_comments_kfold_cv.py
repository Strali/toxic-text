import pandas as pd
from sklearn.model_selection import KFold

import keras.backend as K
from keras.preprocessing import text, sequence
from keras.models import Model
from keras.layers import Activation, Dropout, Flatten, multiply
from keras.layers import Lambda, Permute, RepeatVector
from keras.layers import Bidirectional, Dense, Embedding, Input, CuDNNGRU, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint


def build_model():
    embedding_dim = 128
    hidden_units = 128

    input_placeholder = Input(shape=(MAX_LENGHT, ))
    x = Embedding(input_dim=MAX_NUM_WORDS,
                  output_dim=embedding_dim)(input_placeholder)
    x = Bidirectional(CuDNNGRU(units=hidden_units,
                               return_sequences=True),
                      merge_mode='mul')(x)

    att = TimeDistributed(Dense(1, activation='tanh'))(x)
    att = Flatten()(att)
    att = Activation('softmax')(att)
    att = RepeatVector(hidden_units)(att)
    att = Permute([2, 1])(att)

    sent_rep = multiply([x, att])
    sent_rep = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(hidden_units,))(sent_rep)

    sent_rep = Dropout(0.1)(sent_rep)
    sent_rep = Dense(hidden_units, activation='elu')(sent_rep)
    sent_rep = Dropout(0.1)(sent_rep)
    sent_rep = Dense(int(hidden_units/2), activation='elu')(sent_rep)
    sent_rep = Dropout(0.1)(sent_rep)
    probs = Dense(6, activation='sigmoid')(sent_rep)

    model = Model(input=input_placeholder, output=probs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


MAX_NUM_WORDS = 20000
MAX_LENGHT = 128

BATCH_SIZE = 64
N_EPOCHS = 5
N_FOLDS = 5

kf = KFold(n_splits=N_FOLDS, shuffle=True)

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')

train_comments = train['comment_text'].fillna('MISSING').values
class_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_targets = train[class_list].values
test_comments = test['comment_text'].fillna('MISSING').values

tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(list(list(train_comments)))
train_comments_tokenized = tokenizer.texts_to_sequences(train_comments)
test_comments_tokenized = tokenizer.texts_to_sequences(test_comments)

X_train = sequence.pad_sequences(train_comments_tokenized, MAX_LENGHT)
X_test = sequence.pad_sequences(test_comments_tokenized, MAX_LENGHT)

file_path = 'weights_base.best.hdf5'
CP_callback = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,
                              save_best_only=True, mode='min')
ES_callback = EarlyStopping(monitor='val_loss', mode='min', patience=20)
callbacks = [CP_callback, ES_callback]

model = build_model()
model.summary()
# config = model.get_config()
i = 1

for (train_index, val_index) in kf.split(X_train):
    print('Fitting folds %d' % i)

    fold_model = Model.from_config(model.get_config())
    fold_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    file_path = './weights/weights_base_fold%d.best.hdf5' % i
    CP_callback = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='min')
    ES_callback = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    callbacks = [CP_callback, ES_callback]

    fold_model.fit(X_train[train_index], train_targets[train_index],
                   batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                   validation_data=(X_train[val_index], train_targets[val_index]),
                   callbacks=callbacks)

    fold_model.load_weights(file_path)
    y_test = fold_model.predict(X_test)
    if i == 1:
        sample_submission[class_list] = y_test
    else:
        sample_submission[class_list] += y_test
    i += 1

sample_submission[class_list] /= N_FOLDS
sample_submission.to_csv('submissions/attention_bigru_kfold_high_units.csv', index=False)
