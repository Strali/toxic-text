import numpy as np
import pandas as pd

from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D
from keras.models import Sequential

from preprocessing import TextPreprocessor

data = pd.read_csv('./data/train.csv', nrows=10000)

TextPreprocessor.text_normalization(data)
comments = data['comment_text']
labels = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
print(comments[0:5])

txt = ''
for cmnt in comments:
    for s in cmnt:
        txt += s

all_chars = set(txt)
print('Total chars: ', len(all_chars))
char_indices = dict((c, i) for i, c in enumerate(all_chars))
indices_char = dict((i, c) for i, c in enumerate(all_chars))

maxlen = 500
vocab_size = len(all_chars)
X = np.zeros((len(comments), maxlen, vocab_size), dtype=np.int32)

for i, cmnt in enumerate(comments):
    counter = 0
    cmnt_array = np.zeros((maxlen, vocab_size))
    chrs = list(cmnt.replace(' ', ''))
    for c in chrs:
        if counter >= maxlen:
            pass
        else:
            char_array = np.zeros(vocab_size, dtype=np.int32)
            ix = char_indices[c]
            char_array[ix] = 1
            cmnt_array[counter, :] = char_array
            counter += 1

    X[i, :, :] = cmnt_array

filter_numbers = [64, 128]
filter_sizes = [3, 3]

model = Sequential()
model.add(Conv2D(32, 5, strides=1, input_shape=(maxlen, vocab_size, 1)))
for n in range(len(filter_numbers)):
    model.add(Conv2D(filter_numbers[n], filter_sizes[n], strides=1, padding='same'))
    model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(X[..., None], labels, batch_size=256, epochs=5, validation_split=0.1)
