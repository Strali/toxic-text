import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import concatenate, Dropout, GlobalMaxPool1D, SpatialDropout1D, multiply
from keras.layers import Activation, Lambda, Permute, Reshape, RepeatVector, TimeDistributed
from keras.layers import Bidirectional, CuDNNGRU, Dense, Embedding, Input

from utils import get_embeddings
from utils import shuffle_data


class ToxicClassifier(object):
    """Class for classifying the toxicity of a sentence."""
    def __init__(self, embedding_dim, num_timesteps, word_index, weight_path,
                 use_aux_input=False, average_attention=False,
                 use_ft=False, visualize=False):
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps

        self.weight_path = weight_path

        self.average_attention = average_attention
        self.use_aux_input = use_aux_input
        self.use_ft = use_ft
        self.visualize = visualize

        self.CLASS_LIST = ['toxic', 'severe_toxic', 'obscene',
                           'threat', 'insult', 'identity_hate']

    def set_input_and_labels(self, X_train, y_train, X_aux=None):
        """Set training input and -labels.
        
        Parameters:
        -----------
        X_train : Array of input features.
        y_train : Array of output labels.
        X_aux : Optional array of auxilliary inputs, i.e. engineered features.

        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_aux = X_aux

    def set_sample_sentence(self, sample_text, sample_sequence, sample_aux=None):
        """Set sample sentence and variables to store attention activations.

        Parameters:
        -----------
        sample_text : Preprocessed text of sample.
        sample_sequence : Padded and tokenized representation of sample text.
        sample_aux : Optional auxilliary input for sample, i.e. engineered features.
        """
        self.sample_text = sample_text
        self.sample_sequence = sample_sequence
        self.sample_aux = sample_aux
        self.sample_length = len(self.sample_text.split(' '))
        if self.visualize:
            self.attention_history = np.zeros((1, self.sample_length))

    def _attention_3d_block(self, inputs):
        """Return attention vector evaluated over input. If SINGLE_ATTENTION_VECTOR
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
        a = Reshape((input_dim, self.num_timesteps))(a)
        a = Dense(self.num_timesteps, activation='softmax')(a)

        if self.average_attention:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)

        a_probs = Permute((2, 1), name='attention_vec')(a)
        output_attention = multiply([inputs, a_probs], name='attention_mul')

        return output_attention

    def build_model(self, word_index):
        """Return a compiled Keras model for sentence classification.

        Parameters:
        -----------
        word_index : List of tokens in input data
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

        lstm_input = Input(shape=(self.num_timesteps, ), name='lstm_input')
        embedding_matrix = get_embeddings(word_index=word_index,
                                          embedding_dim=self.embedding_dim,
                                          use_ft_embeddings=self.use_ft)
        x = Embedding(len(word_index) + 1,
                      self.embedding_dim,
                      weights=[embedding_matrix],
                      input_length=self.num_timesteps,
                      trainable=False)(lstm_input)

        x = SpatialDropout1D(0.3)(x)
        x = Bidirectional(CuDNNGRU(units=hidden_units_1,
                                   return_sequences=True))(x)
        x = SpatialDropout1D(0.3)(x)
        x = Bidirectional(CuDNNGRU(units=hidden_units_2,
                                   return_sequences=True))(x)
        x = TimeDistributed(Activation('tanh'))(x)

        attention = self._attention_3d_block(inputs=x)
        dense_input = GlobalMaxPool1D()(attention)

        if self.use_aux_input:
            aux_input = Input(shape=(3, ), name='aux_input')
            dense_input = concatenate([dense_input, aux_input])

        dense = Dropout(0.2)(dense_input)
        dense = Dense(dense_units_1, activation='elu')(dense)
        dense = Dropout(0.2)(dense)
        dense = Dense(dense_units_2, activation='elu')(dense)
        dense = Dropout(0.2)(dense)
        probs = Dense(6, activation='sigmoid')(dense)

        if self.use_aux_input:
            self.model = Model(inputs=[lstm_input, aux_input], output=probs)
        else:
            self.model = Model(inputs=lstm_input, output=probs)

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.attention_layer_model = Model(inputs=self.model.input,
                                           outputs=self.model.get_layer('attention_mul').output)

    def load_best_weights(self):
        """Load best weights into model."""
        self.model.load_weights(self.weight_path)

    def train(self, max_epochs, batch_size, callbacks):
        """Train model, using specified epoch number, batch_size and callbacks.
        
        Parameters:
        -----------
        max_epochs : The maximum number of epochs to train for.
        batch_size : Batch size used during training, i.e. the number of sentences.
        callbacks : List of Keras callbacks.

        """
        if self.use_aux_input:
            _X_train, _y_train, _X_aux = shuffle_data(self.X_train, self.y_train, self.X_aux)
        else:
            _X_train, _y_train = shuffle_data(self.X_train, self.y_train)

        if self.visualize:
            for epoch in range(max_epochs):
                if self.use_aux_input:
                    self.model.fit([_X_train, _X_aux], _y_train, 
                                   batch_size=batch_size,
                                   epochs=epoch + 1, 
                                   initial_epoch=epoch,
                                   validation_split=0.1, 
                                   callbacks=callbacks)
                    attention_output = self.attention_layer_model.predict(
                        [self.sample_sequence.reshape(1, self.num_timesteps),
                         self.sample_aux.reshape(1, 3)], batch_size=1)
                else:
                    self.model.fit(_X_train, _y_train, 
                                   batch_size=batch_size,
                                   epochs=epoch + 1,
                                   initial_epoch=epoch,
                                   validation_split=0.1,
                                   callbacks=callbacks)
                    attention_output = self.attention_layer_model.predict(
                        self.sample_sequence .reshape(1, self.num_timesteps),  batch_size=1)

                self.attention_history = np.append(
                    self.attention_history, 
                    [attention_output[0, -self.sample_length:, 0]],
                    axis=0)
        else:
            if self.use_aux_input:
                self.model.fit([_X_train, _X_aux], _y_train,
                               batch_size=batch_size,
                               epochs=max_epochs,
                               validation_split=0.1,
                               callbacks=callbacks)
            else:
                self.model.fit(_X_train, _y_train, 
                               batch_size=batch_size,
                               epochs=max_epochs,
                               validation_split=0.1,
                               callbacks=callbacks)
        print('Training done')

    def predict_on_dataset(self, data, aux_input=None):
        """Predict on an entire dataset at once using trained model.
        
        Parameters:
        -----------
        data : Numpy array containing input data.
        aux_input : Optional auxilliary input (i.e. engineered features).
        """
        self.load_best_weights
        if aux_input is not None:
            try:
                assert self.use_aux_input
            except AssertionError:
                print('ERROR: Unexpexcted auxilliary input passed to predict function')
                exit
            preds = self.model.predict([data, aux_input])
        else:
            preds = self.model.predict(data)

        return preds

    def predict_sample_output(self):
        """Predict on a single sample text using trained model."""
        self.load_best_weights
        if self.use_aux_input:
            pred = self.model.predict([self.sample_sequence.reshape(1, self.num_timesteps),
                                       self.sample_aux.reshape(1, 3)],
                                      batch_size=1)
        else:
            pred = self.model.predict(self.sample_sequence.reshape(1, self.num_timesteps),
                                      batch_size=1)

        return pred
