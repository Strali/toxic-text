import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def get_callbacks(*callback_params):
    """Return a list of Keras callbacks.
    NOTE: Assumes callback parameters are given in the order
        "ModelCheckpoint", "EarlyStopping", "TensorBoard".
    """

    callback_list = []

    CP_callback = ModelCheckpoint(**callback_params[0])
    callback_list.append(CP_callback)
    ES_callback = EarlyStopping(**callback_params[1])
    callback_list.append(ES_callback)

    if len(callback_params) == 3:
        TB_callback = TensorBoard(**callback_params[2])
        callback_list.append(TB_callback)

    return callback_list


def get_embeddings(word_index, embedding_dim=300, use_ft_embeddings=False):
    """
    Return pre-trained word embeddings for words in the input corpus. OOV
    words will be encoded using a N(0, 1) random distribution.

    Parameters:
    -----------
    word_index : List of all tokens (words) in the corpus.
    embedding_dim : (Scalar) dimension of the embedding space.
    use_ft : Boolean, whether to use word vectors pretrained using fasttext.

    NOTE: GloVe embeddings reqiure embedding_dim==300. Fasttext embeddings
        can have embedding_dim==100 or embedding_dim==300

    Returns:
    --------
    embedding_matrix : A (num_words + 1, embedding_dim) matrix of word embeddings

    """

    embeddings_index = {}
    if use_ft_embeddings:
        print('Using fasttext word embeddings...')
        embedding_file = './data/word_embeddings/ft_skipgram_300d.txt'
    else:
        print('Using GloVe embeddings...')
        assert embedding_dim == 300
        embedding_file = './data/word_embeddings/glove.42B.300d.txt'

    print('Creating word embeddings from file %s' % embedding_file)
    f = open(embedding_file, encoding='utf-8')

    n = 1
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError as e:
            print('Error on line', n, ': ', e)
        n += 1
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(0, 1, embedding_dim)
    print('Loaded embedding matrix\n')

    return embedding_matrix


def get_sentence(index, classes, comments):
    """Return the sentence and corresponding labels at index position"""

    sentence_text = comments['comment_text'].values[index]
    print(sentence_text)
    sentence_label = comments[classes].values[index]
    return sentence_text, sentence_label


def get_train_comments():
    """Return all raw comments in training data"""
    train = pd.read_csv('./data/train.csv')
    return train['comment_text'].fillna('MISSING').values


def make_aux_submission(model, X_test, X_aux, class_list, weight_path, output_path):
    """
    Predict on test data using supplied (trained) model and create Kaggle
        submission file using predictions.

    Parameters:
    -----------
    model : A trained Keras model.
    X_test : Array of (preprocessed) test data.
    X_aux : Array of auxilliary inputs (i.e. engineered features)
    class_list : 1-D array of all classes in output.
    weight_path : Path to model weights.
    output_path : Path to which submission is saved.

    """

    model.load_weights(weight_path)
    y_test = model.predict([X_test, X_aux])

    sample_submission = pd.read_csv('./data/sample_submission.csv')
    sample_submission[class_list] = y_test
    sample_submission.to_csv(output_path, index=False)


def make_submission(model, X_test, class_list, weight_path, output_path):
    """
    Predict on test data using supplied (trained) model and create Kaggle
        submission file using predictions.

    Parameters:
    -----------
    model : A trained Keras model.
    X_test : Array of (preprocessed) test data.
    class_list : 1-D array of all classes in output.
    weight_path : Path to model weights.
    output_path : Path to which submission is saved.

    """

    model.load_weights(weight_path)
    y_test = model.predict(X_test)

    sample_submission = pd.read_csv('./data/sample_submission.csv')
    sample_submission[class_list] = y_test
    sample_submission.to_csv(output_path, index=False)


def shuffle_data(features, labels):
    """Return features and labels in random order"""

    assert features.shape[0] == labels.shape[0]
    p = np.random.permutation(features.shape[0])

    return features[p], labels[p]


def shuffle_extended_data(features, aux, labels):
    """Return features and labels in random order"""

    assert features.shape[0] == labels.shape[0]
    p = np.random.permutation(features.shape[0])

    return features[p], aux[p], labels[p]


def visualize_attention(attention_vector, input_text):
    """
    Plot attention activations for input text over a number of epochs.

    Parameters:
    -----------
    attention_vector : Array of attention activations for each time input_text
        has beed seen during training (i.e. once per epoch).
    input_text : The text string for which attention is to be shown.

    """

    input_split = input_text.split(' ')
    input_length = len(input_split)
    total_epochs = attention_vector.shape[0] - 1

    f = plt.figure(figsize=(8.5, int((total_epochs + 2)/2)))
    ax = f.add_subplot(1, 1, 1)

    activation_map = attention_vector[1:, :]
    i = ax.imshow(activation_map,
                  interpolation='nearest',
                  cmap=plt.get_cmap('YlOrRd'))
    cbaxes = f.add_axes([0.2, 0.93, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Attention value', labelpad=2)

    ax.set_yticklabels('')
    ax.set_xticklabels('')
    x_ticks = np.linspace(0, input_length-1, num=input_length)
    y_ticks = np.linspace(1, total_epochs, total_epochs)

    ax.set_xticks(x_ticks, minor=True)
    ax.set_yticks(y_ticks-0.5, minor=False)

    ax.set_xticklabels(input_split, minor=True, rotation=90)
    ax.set_yticklabels(y_ticks, minor=False)

    plt.show()
