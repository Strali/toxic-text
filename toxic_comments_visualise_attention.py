"""

TODO: Fix so that the correct sentence representation is used for visualising attention
    when shuffling training data
"""
import argparse
import datetime

import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import concatenate, Dropout, GlobalMaxPool1D, SpatialDropout1D, multiply
from keras.layers import Activation, Lambda, Permute, Reshape, RepeatVector, TimeDistributed
from keras.layers import Bidirectional, CuDNNGRU, Dense, Embedding, Input

from preprocessing import load_and_tokenize
from utils import get_embeddings, get_callbacks
from utils import make_aux_submission, make_submission
from utils import shuffle_data, shuffle_extended_data
from utils import visualize_attention


def attention_3d_block(inputs, num_timesteps, average_attention_temporaly=False):
    """
    Return attention vector evaluated over input. If SINGLE_ATTENTION_VECTOR
    argument is given a temporal mean is taken over the time_step dimension.

    Parameters:
    -----------
    inputs : A tensor of shape (batch_size, time_steps, input_dim).
        Time_steps is represented by the input length, i.e. the number of tokens,
        while input_dim is the number of nodes in the previous nn layer.

    Returns:
    --------
    output_attention :  A tensor of shape (batch_size, time_steps, input_dim),
        representing the attention given to each input token.

    """

    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, num_timesteps))(a)
    a = Dense(num_timesteps, activation='softmax')(a)

    if average_attention_temporaly:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention = multiply([inputs, a_probs], name='attention_mul')

    return output_attention


def build_model(embedding_dim, num_timesteps, word_index,
                use_aux_input=False, average_attention=False, use_ft=False):
    """
    Return a compiled Keras model for sentence classification.

    Parameters:
    -----------
    embedding_dim : (Scalar) dimension of word vector embeddings.
    word_index : List of tokens in input data
    use_ft : Boolean, whether to use word vectors pretrained using fasttext.

    NOTE: GloVe embeddings reqiure embedding_dim==300. Fasttext embeddings
        can have embedding_dim==100 or embedding_dim==300


    Returns:
    --------
    model : A compiled Keras model for predicting six types of toxicity
        in a sentencee.
    attention_layer_model : A Keras model for extracting the attention
        layer output.

    """

    hidden_units_1 = 128
    hidden_units_2 = 128
    dense_units_1 = 256
    dense_units_2 = 128

    lstm_input = Input(shape=(num_timesteps, ), name='lstm_input')
    embedding_matrix = get_embeddings(word_index,
                                      embedding_dim,
                                      use_ft_embeddings=use_ft)
    x = Embedding(len(word_index) + 1,
                  embedding_dim,
                  weights=[embedding_matrix],
                  input_length=num_timesteps,
                  trainable=False)(lstm_input)

    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNGRU(units=hidden_units_1,
                               return_sequences=True))(x)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(CuDNNGRU(units=hidden_units_2,
                               return_sequences=True))(x)
    x = TimeDistributed(Activation('tanh'))(x)

    attention = attention_3d_block(inputs=x,
                                   num_timesteps=num_timesteps,
                                   average_attention_temporaly=average_attention)
    dense_input = GlobalMaxPool1D()(attention)

    if use_aux_input:
        aux_input = Input(shape=(3, ), name='aux_input')
        dense_input = concatenate([dense_input, aux_input])

    dense = Dropout(0.2)(dense_input)
    dense = Dense(dense_units_1, activation='elu')(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(dense_units_2, activation='elu')(dense)
    dense = Dropout(0.2)(dense)
    probs = Dense(6, activation='sigmoid')(dense)

    if use_aux_input:
        model = Model(inputs=[lstm_input, aux_input], output=probs)
    else:
        model = Model(inputs=lstm_input, output=probs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    attention_layer_model = Model(inputs=model.input,
                                  outputs=model.get_layer('attention_mul').output)

    return model, attention_layer_model


def main(args):
    MAX_NUM_WORDS = 60000
    MAX_LENGTH = 120
    EMBEDDING_DIM = 300

    MAX_EPOCHS = 10
    BATCH_SIZE = 256
    SENTENCE_NUM = 32

    AVERAGE_ATTENTION = False

    TRAIN = args.train
    USE_EXTRA_FEATURES = args.more_features
    USE_EXTENDED_DATA = args.extended_data
    MAKE_SUBMISSION = args.submit
    VISUALISE_ATTENTION = args.visualise
    USE_FASTTEXT = args.fasttext

    CLASS_LIST = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    if USE_EXTRA_FEATURES:
        X_train, X_aux, y_train, X_test, test_aux, word_index, sample_text, sample_target = \
            load_and_tokenize(CLASS_LIST,
                              MAX_NUM_WORDS,
                              MAX_LENGTH,
                              track_index=SENTENCE_NUM,
                              use_extra_features=USE_EXTRA_FEATURES,
                              use_extended_data=USE_EXTENDED_DATA)
    else:
        X_train, y_train, X_test, word_index, sample_text, sample_target = \
            load_and_tokenize(CLASS_LIST,
                              MAX_NUM_WORDS,
                              MAX_LENGTH,
                              track_index=SENTENCE_NUM,
                              use_extra_features=USE_EXTRA_FEATURES,
                              use_extended_data=USE_EXTENDED_DATA)

    sample_sequence = X_train[SENTENCE_NUM]
    if USE_EXTRA_FEATURES:
        sample_aux = X_aux[SENTENCE_NUM]
    sentence_length = len(sample_text.split(' '))
    attention_history = np.zeros((1, sentence_length))

    model, attention_layer = build_model(embedding_dim=EMBEDDING_DIM,
                                         num_timesteps=MAX_LENGTH,
                                         word_index=word_index,
                                         use_aux_input=USE_EXTRA_FEATURES,
                                         average_attention=AVERAGE_ATTENTION,
                                         use_ft=USE_FASTTEXT)
    model.summary()

    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d%H%M')
    LOG_PATH = './logs/' + now
    WEIGHT_SAVE_PATH = 'weights_base.best.hdf5'
    SUBMISSION_SAVE_PATH = './submissions/submission_' + now + '.csv'
    ES_PATIENCE = 5
    TB_HIST_FREQ = 0
    TB_WRITE_GRAPH = True

    ckpt_params = {'filepath': WEIGHT_SAVE_PATH, 'verbose': 1,
                   'save_best_only': True, 'save_weights_only': True}
    es_params = {'patience': ES_PATIENCE}
    tb_params = {'log_dir': LOG_PATH, 'histogram_freq': TB_HIST_FREQ,
                 'write_graph': TB_WRITE_GRAPH, 'batch_size': BATCH_SIZE,
                 'embeddings_freq': MAX_EPOCHS + 1}

    callbacks = get_callbacks(ckpt_params, es_params, tb_params)

    if TRAIN:
        if USE_EXTRA_FEATURES:
            X_train, X_aux, y_train = shuffle_extended_data(X_train, X_aux, y_train)
        else:
            X_train, y_train = shuffle_data(X_train, y_train)

        if VISUALISE_ATTENTION:
            for epoch in range(MAX_EPOCHS):
                if USE_EXTRA_FEATURES:
                    model.fit([X_train, X_aux], y_train, batch_size=BATCH_SIZE,
                              epochs=epoch + 1, initial_epoch=epoch,
                              validation_split=0.1, callbacks=callbacks)
                    attention_output = attention_layer.predict(
                        [sample_sequence.reshape(1, MAX_LENGTH),
                         sample_aux.reshape(1, 3)],
                        batch_size=1)
                else:
                    model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                              epochs=epoch + 1, initial_epoch=epoch,
                              validation_split=0.1, callbacks=callbacks)
                    attention_output = attention_layer.predict(
                        sample_sequence .reshape(1, MAX_LENGTH),  batch_size=1)

                attention_history = np.append(
                    attention_history, [attention_output[0, -sentence_length:, 0]], axis=0)
        else:
            if USE_EXTRA_FEATURES:
                model.fit([X_train, X_aux], y_train, batch_size=BATCH_SIZE,
                          epochs=MAX_EPOCHS, validation_split=0.1,
                          callbacks=callbacks)
            else:
                model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                          epochs=MAX_EPOCHS, validation_split=0.1,
                          callbacks=callbacks)
        print('Training done')
        if USE_EXTRA_FEATURES:
            pred = model.predict([sample_sequence.reshape(1, MAX_LENGTH),
                                  sample_aux.reshape(1, 3)],
                                 batch_size=1)
        else:
            pred = model.predict(sample_sequence.reshape(1, MAX_LENGTH), batch_size=1)

        print('Original sentence: ', sample_text)
        print('Actual label: ', sample_target)
        print('Model prediction :', pred[0, :])

    if VISUALISE_ATTENTION:
        visualize_attention(attention_history, sample_text)

    if MAKE_SUBMISSION:
        print('Loading best weights and predicting on test data\n')
        if USE_EXTRA_FEATURES:
            make_aux_submission(model, X_test, test_aux, CLASS_LIST,
                                WEIGHT_SAVE_PATH, SUBMISSION_SAVE_PATH)
        else:
            make_submission(model, X_test, CLASS_LIST,
                            WEIGHT_SAVE_PATH, SUBMISSION_SAVE_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Toxicity model parameters.')
    parser.add_argument('-t',
                        '--train',
                        action='store_true',
                        help='Retrain the model')
    parser.add_argument('-s',
                        '--submit',
                        action='store_true',
                        help='Make Kaggle submission after training')
    parser.add_argument('-v',
                        '--visualise',
                        action='store_true',
                        help='Visualise attention activations during training')
    parser.add_argument('-e',
                        '--extended_data',
                        action='store_true',
                        help='Use extended data for training')
    parser.add_argument('-f',
                        '--fasttext',
                        action='store_true',
                        help='Use fasttext embeddings instead of GloVe')
    parser.add_argument('-m',
                        '--more_features',
                        action='store_true',
                        help='Use engineered features as auxilarry model input')
    args = parser.parse_args()

    main(args)
