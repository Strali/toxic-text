import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class RocAucEvaluation(Callback):
    """Evaluate ROC AUC on validation data at epoch end.

    Parameters
    ----------
    interval : Number of epochs between ROC evaluations, default 1.

    """

    def __init__(self, interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.roc = 0.0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            if len(self.validation_data) == 5:
                y_pred = self.model.predict([self.validation_data[0], self.validation_data[1]],
                                            verbose=0)
                score = roc_auc_score(self.validation_data[2], y_pred)
            else:
                y_pred = self.model.predict(self.validation_data[0], verbose=0)
                score = roc_auc_score(self.validation_data[1], y_pred)

            print('ROC AUC score on validation data - epoch: {:d} - ROC AUC: {:.6f}'
                  .format(epoch + 1, score))
            if score > self.roc:
                print('Best ROC score increased from {:.6f} to {:.6f}\n'
                      .format(self.roc, score))
                self.roc = score
            else:
                print('ROC score did not improve\n')


class TrainValTensorBoard(TensorBoard):
    """Enhance Tensorboard callback to display training and validation
        metrics on a single graph for easy comparison.

    Parameters
    ----------
    log_dir : String specifying base path to log directory.
    **kwargs : Keyword arguments to TensorBoard callback.
    """

    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = log_dir + '/training'
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = log_dir + '/validation'

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def get_callbacks(*callback_params):
    """Return a list of Keras callbacks.
    NOTE: Assumes callback parameters are given in the order
        "CyclicLR", "ModelCheckpoint", "EarlyStopping", "TensorBoard".
    """

    callback_list = []

    CLR_callback = CyclicLR(base_lr=0.0001, max_lr=0.005, step_size=30000,
                            mode='triangular')
    callback_list.append(CLR_callback)
    CP_callback = ModelCheckpoint(**callback_params[1])
    callback_list.append(CP_callback)
    ES_callback = EarlyStopping(**callback_params[2])
    callback_list.append(ES_callback)

    if len(callback_params) == 3:
        TB_callback = TrainValTensorBoard(**callback_params[3])
        callback_list.append(TB_callback)

    ROC_callback = RocAucEvaluation(interval=1)
    callback_list.append(ROC_callback)

    return callback_list


def get_embeddings(word_index, embedding_dim=300,
                   use_ft_embeddings=False, use_skipgram=True):
    """
    Return pre-trained word embeddings for words in the input corpus. OOV
    words will be encoded using a N(0, 1) random distribution.

    Parameters
    ----------
    word_index : List of all tokens (words) in the corpus.
    embedding_dim : (Scalar) dimension of the embedding space.
    use_ft : Boolean, whether to use word vectors pretrained using fasttext.
    use_skipgram : Boolean, whether to use fasttext skipgram word vectors.
        If false, cbow model word vectors will be used instead.

    NOTE: GloVe embeddings reqiure embedding_dim==300. Fasttext embeddings
        can have embedding_dim==100 or embedding_dim==300

    Returns
    -------
    embedding_matrix : A (num_words + 1, embedding_dim) matrix of word embeddings

    """

    embeddings_index = {}
    if use_ft_embeddings:
        if use_skipgram:
            print('Using fasttext skipgram embeddings...')
            embedding_file = './data/word_embeddings/ft_skipgram_300d.txt'
        else:
            print('Using fasttext cbow embeddings...')
            embedding_file = './data/word_embeddings/ft_cbow_300d.txt'
    else:
        print('Using GloVe embeddings...')
        assert embedding_dim == 300
        embedding_file = './data/word_embeddings/glove.840B.300d.txt'

    print('Creating word embeddings from file %s' % embedding_file)
    f = open(embedding_file, encoding='utf-8')

    n = 1
    for line in tqdm(f):
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
    oov_count = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            oov_count += 1
            embedding_matrix[i] = np.random.normal(0, 1, embedding_dim)
    print('Loaded embedding matrix')
    print('%i (%.2f%%) oov words found in data\n' %
          (oov_count, 100 * (oov_count / len(word_index))))

    return embedding_matrix


def get_mean_log_loss(true, pred, eps=1e-15):
    """Return log loss of input"""
    return log_loss(true, pred, eps)


def get_sentence(index, classes, comments):
    """Return the sentence and corresponding labels at index position"""

    sentence_text = comments['comment_text'].values[index]
    sentence_label = comments[classes].values[index]
    return sentence_text, sentence_label


def get_train_comments():
    """Return all raw comments in training data"""
    train = pd.read_csv('./data/train.csv')
    return train['comment_text'].fillna('MISSING').values


def get_toxicity_classes(preds, threshold=0.6, classes=None):
    """Print detected classes of toxicity in a sentence based on threshold.

    Parameters
    ----------
    preds : Probabilities of each type of toxicity
    threshold : Threshold probability for when a type of toxicity is present
    classes : List of all toxicity types

    """
    if classes is None:
        classes = ['toxic', 'severe_toxic', 'obscene',
                   'threat', 'insult', 'identity_hate']

    if sum(preds > threshold) > 0:
        present_classes = [classes[i] for i in range(len(classes))
                           if preds[i] > threshold]
        return present_classes
    else:
        return ['not_toxic']


def make_aux_submission(model, X_test, X_aux,
                        class_list, weight_path, output_path,
                        post_process=False):
    """
    Predict on test data using supplied (trained) model and create Kaggle
        submission file using predictions.

    Parameters
    ----------
    model : A trained Keras model.
    X_test : Array of (preprocessed) test data.
    X_aux : Array of auxilliary inputs (i.e. engineered features)
    class_list : 1-D array of all classes in output.
    weight_path : Path to model weights.
    output_path : Path to which submission is saved.

    """
    model.load_weights(weight_path)
    y_test = model.predict([X_test, X_aux])

    if post_process:
        y_test = expit(logit(y_test) - 0.5)

    sample_submission = pd.read_csv('./data/sample_submission.csv')
    sample_submission[class_list] = y_test
    sample_submission.to_csv(output_path, index=False)


def make_submission(model, X_test, class_list, weight_path, output_path):
    """
    Predict on test data using supplied (trained) model and create Kaggle
        submission file using predictions.

    Parameters
    ----------
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


def print_toxicity_report(preds, threshold=0.6, classes=None):
    """Print detected classes of toxicity in a sentence based on threshold.

    Parameters
    ----------
    preds : Probabilities of each type of toxicity
    threshold : Threshold probability for when a type of toxicity is present
    classes : List of all toxicity types

    """
    if classes is None:
        classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    if sum(preds > threshold) > 0:
        precent_classes = [classes[i] for i in range(len(classes))
                           if preds[i] > threshold]
        toxicity_levels = [preds[i] for i in range(len(classes))
                           if preds[i] > threshold]
        print('Based on a toxicity threshold of ', str(threshold), 'the sentence',
              'is predicted to contain toxic language of the following types;')
        for i in range(len(precent_classes)):
            print('\t- %s, with probability %.2f' % (precent_classes[i], toxicity_levels[i]))
    else:
        print('Based on a toxicity threshold of %.1f, ' % threshold,
              'the sentence is predicted to contain no toxic language!')


def save_training_predictions(model, X_train, X_aux, class_list, weight_path, output_path):
    """
    Predict on test data using supplied (trained) model and create Kaggle
        submission file using predictions.

    Parameters
    ----------
    model : A trained Keras model.
    X_test : Array of (preprocessed) test data.
    X_aux : Array of auxilliary inputs (i.e. engineered features)
    class_list : 1-D array of all classes in output.
    weight_path : Path to model weights.
    output_path : Path to which submission is saved.

    """
    model.load_weights(weight_path)
    y_test = model.predict([X_train, X_aux])

    y_df = pd.DataFrame(data=y_test, columns=class_list)
    y_df.to_csv(output_path, index=False)


def shuffle_data(features, labels, aux=None):
    """Return features and labels in random order"""

    assert features.shape[0] == labels.shape[0]
    p = np.random.permutation(features.shape[0])

    if aux is not None:
        return features[p], labels[p], aux[p]
    else:
        return features[p], labels[p]


def visualise_attention(attention_vector, input_text, num_epochs=None):
    """Plot attention activations for input text over a number of epochs.

    Parameters
    ----------
    attention_vector : Array of attention activations for each time input_text
        has beed seen during training (i.e. once per epoch).
    input_text : The text string for which attention is to be shown.

    """

    input_split = input_text.split(' ')
    input_length = len(input_split)
    total_epochs = attention_vector.shape[0] - 1

    f = plt.figure(figsize=(8.5, int((total_epochs + 2) / 2)))
    ax = f.add_subplot(1, 1, 1)

    if num_epochs is None:
        activation_map = attention_vector[1:, :]
    else:
        num_rows = np.minimum(total_epochs, num_epochs)
        activation_map = attention_vector[-num_rows:, :]
    i = ax.imshow(activation_map,
                  interpolation='nearest',
                  cmap=plt.get_cmap('YlOrRd'))
    cbaxes = f.add_axes([0.2, 0.93, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Attention value', labelpad=2)

    ax.set_yticklabels('')
    ax.set_xticklabels('')
    x_ticks = np.linspace(0, input_length - 1, num=input_length)
    y_ticks = np.linspace(1, total_epochs, total_epochs)

    ax.set_xticks(x_ticks, minor=True)
    ax.set_yticks(y_ticks - 0.5, minor=False)

    ax.set_xticklabels(input_split, minor=True, rotation=90)
    ax.set_yticklabels(y_ticks, minor=False)

    plt.show()


def visualise_attention_with_text(attention_vector, input_text,
                                  preds, present_classes,
                                  target=None, labels=None):
    """Visualise the attenton vector during the last epoch of training.

    Parameters
    ----------
    attention_vector : Array of attention activations for each time input_text
        has beed seen during training (i.e. once per epoch).
    input_text : The text string for which attention is to be shown.
    preds : Class predictions made by classification model.
    target : (Optional) class labels for the input sentence.

    """
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    title = 'Word attention visualisation'
    plt.title(title, fontsize=18)
    ttl = ax.title
    ttl.set_position([0.5, 1.05])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    input_split = input_text.split(' ')
    input_array = np.array(input_split).reshape((len(input_split), 1))
    attention_map = attention_vector.reshape((attention_vector.shape[0], 1))
    cbar_kws = {'label': 'Percentage of total attention',
                'orientation': 'horizontal'}
    sns.heatmap(attention_map, annot=input_array, fmt='', cmap='YlOrRd',
                cbar_kws=cbar_kws, linewidths=0.30, ax=ax)

    if target is not None:

        txt = 'Actual label: ' + str(target) + \
            '\nCorresponding classes: ' + str(labels) + \
            '\nPredicted classes: ' + str(present_classes)
        # '\nModel prediction: %.2f' % (preds.round(2)) + \

    else:
        txt = '\nModel prediction: ' + str(preds.round(2)) + \
            '\nPredicted toxicity: ' + str(present_classes)

    fig.text(0.5, 0.05, txt, ha='center')
    fig.set_size_inches(5, 10, forward=True)

    plt.show()
