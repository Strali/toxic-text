import pandas as pd

import keras.backend as K
from keras.preprocessing import text, sequence
from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, Flatten, GlobalMaxPool1D, multiply
from keras.layers import Lambda, Permute, RepeatVector
from keras.layers import Bidirectional, Dense, Embedding, Input, CuDNNLSTM, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint


def build_model():
    embedding_dim = 128
    hidden_units = 128

    input_placeholder = Input(shape=(MAX_LENGHT, ))
    x = Embedding(input_dim=MAX_NUM_WORDS,
                  output_dim=embedding_dim)(input_placeholder)
    x = Bidirectional(CuDNNLSTM(units=hidden_units,
                                return_sequences=True))(x)

    att = TimeDistributed(Dense(1, activation='tanh'))(x)
    att = Flatten()(att)
    att = Activation('softmax')(att)
    att = RepeatVector(hidden_units)(att)
    att = Permute([2, 1])(att)

    sent_rep = multiply([x, att])
    sent_rep = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(hidden_units,))(sent_rep)
    '''
    x_2 = Dropout(0.1)(x_2)
    x_2 = Dense(50, activation="relu")(x_2)
    x_2 = Dropout(0.1)(x_2)
    probs = Dense(6, activation='sigmoid')(x_2)
    '''
    sent_rep = Dropout(0.1)(sent_rep)
    sent_rep = Dense(128, activation='relu')(sent_rep)
    sent_rep = Dropout(0.1)(sent_rep)
    probs = Dense(6, activation='sigmoid')(sent_rep)

    model = Model(input=input_placeholder, output=probs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    '''
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=input_placeholder, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    '''
    return model


'''
def build_model():
    embedding_dim = 128
    hidden_units = 50

    #The LSTM  model -  output_shape = (batch, step, hidden)
    model1 = Sequential()
    model1.add(Embedding(input_dim=MAX_NUM_WORDS,
                         output_dim=embedding_dim))
    model1.add(Bidirectional(LSTM(units=hidden_units,
                                  return_sequences=True)))

    #The weight model  - actual output shape  = (batch, step)
    # after reshape : output_shape = (batch, step,  hidden)
    model2 = Sequential()
    model2.add(Dense(input_dim=embedding_dim, output_dim=step))
    model2.add(Activation('softmax'))  # Learn a probability distribution over each  step.
    #Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
    model2.add(RepeatVector(hidden))
    model2.add(Permute(2, 1))

    #The final model which gives the weighted sum:
    model = Sequential()
    # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
    model.add(Merge([model1, model2], 'mul'))
    model.add(TimeDistributedMerge('sum'))  # Sum the weighted elements.

    model.compile(loss='mse', optimizer='sgd')
'''

MAX_NUM_WORDS = 20000
MAX_LENGHT = 100

BATCH_SIZE = 64
N_EPOCHS = 5

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

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

model = build_model()
model.summary()

file_path = 'weights_base.best.hdf5'
CP_callback = ModelCheckpoint(file_path, monitor='val_loss', verbose=1,
                              save_best_only=True, mode='min')
ES_callback = EarlyStopping(monitor='val_loss', mode='min', patience=20)
callbacks = [CP_callback, ES_callback]

model.fit(X_train, train_targets, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
          validation_split=0.1, callbacks=callbacks)

model.load_weights(file_path)
y_test = model.predict(X_test)

sample_submission = pd.read_csv('./data/sample_submission.csv')
sample_submission[class_list] = y_test
sample_submission.to_csv('submissions/attention_lstm_output.csv', index=False)
