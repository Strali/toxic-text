"""
TODO: Add docstrings
"""
import argparse
import datetime

from preprocessing import TextPreprocessor
from toxic_classifier import ToxicClassifier
from utils import get_callbacks, get_mean_log_loss, get_toxicity_report
from utils import make_aux_submission, make_submission
from utils import visualize_attention


def main(args):
    TRAIN = args.train
    USE_AUXILLIARY_INPUT = args.auxilliary_input
    COMBINE_DATA = args.combine_data
    MAKE_SUBMISSION = args.submit
    VISUALISE_ATTENTION = args.visualise
    USE_FASTTEXT = args.fasttext

    MAX_NUM_WORDS = None
    MAX_LENGTH = 120
    EMBEDDING_DIM = 300

    MAX_EPOCHS = 15
    BATCH_SIZE = 256
    SENTENCE_NUM = 32

    TOXICITY_THRESHOLD = 0.6

    AVERAGE_ATTENTION = False

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

    CLASS_LIST = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    txt_prep = TextPreprocessor(max_nb_words=MAX_NUM_WORDS,
                                max_padding_length=MAX_LENGTH,
                                combine_data=COMBINE_DATA,
                                use_extra_features=USE_AUXILLIARY_INPUT)
    if USE_AUXILLIARY_INPUT:
        X_train, X_aux, y_train, X_test, test_aux, word_index, sample_text, sample_target = \
            txt_prep.load_and_tokenize(class_list=CLASS_LIST,
                                       sample_index=SENTENCE_NUM)
    else:
        X_train, y_train, X_test, word_index, sample_text, sample_target = \
            txt_prep.load_and_tokenize(class_list=CLASS_LIST,
                                       sample_index=SENTENCE_NUM)

    tc = ToxicClassifier(embedding_dim=EMBEDDING_DIM,
                         num_timesteps=MAX_LENGTH,
                         word_index=word_index,
                         weight_path=WEIGHT_SAVE_PATH,
                         use_aux_input=USE_AUXILLIARY_INPUT,
                         average_attention=AVERAGE_ATTENTION,
                         use_ft=USE_FASTTEXT,
                         visualize=VISUALISE_ATTENTION)

    if USE_AUXILLIARY_INPUT:
        tc.set_input_and_labels(X_train, y_train, X_aux)
        tc.set_sample_sentence(sample_text, X_train[SENTENCE_NUM], X_aux[SENTENCE_NUM])
    else:
        tc.set_input_and_labels(X_train, y_train)
        tc.set_sample_sentence(sample_text, X_train[SENTENCE_NUM])

    tc.build_model(word_index)
    tc.model.summary()

    if TRAIN:
        tc.train(max_epochs=MAX_EPOCHS,
                 batch_size=BATCH_SIZE,
                 callbacks=callbacks)

        sample_pred = tc.predict_sample_output()

        pred_loss = get_mean_log_loss(sample_target, sample_pred[0, :])
        print('Original sentence: ', sample_text)
        print('Actual label: ', sample_target)
        print('Model prediction :', sample_pred[0, :])
        print('Corresponding log loss: ', pred_loss)
        get_toxicity_report(sample_pred[0, :], TOXICITY_THRESHOLD, CLASS_LIST)

        if VISUALISE_ATTENTION:
            visualize_attention(tc.attention_history, sample_text)

    if MAKE_SUBMISSION:
        print('Loading best weights and predicting on test data\n')
        if USE_AUXILLIARY_INPUT:
            make_aux_submission(tc.model, X_test, test_aux, CLASS_LIST,
                                WEIGHT_SAVE_PATH, SUBMISSION_SAVE_PATH)
        else:
            make_submission(tc.model, X_test, CLASS_LIST,
                            WEIGHT_SAVE_PATH, SUBMISSION_SAVE_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Toxicity model parameters.')
    parser.add_argument('-a',
                        '--auxilliary_input',
                        action='store_true',
                        help='Use engineered features as auxiliary model input')
    parser.add_argument('-c',
                        '--combine_data',
                        action='store_true',
                        help='Combine training and test data when fitting tokenizer')
    parser.add_argument('-f',
                        '--fasttext',
                        action='store_true',
                        help='Use fasttext embeddings instead of GloVe')
    parser.add_argument('-s',
                        '--submit',
                        action='store_true',
                        help='Make Kaggle submission after training')
    parser.add_argument('-t',
                        '--train',
                        action='store_true',
                        help='Retrain the model')
    parser.add_argument('-v',
                        '--visualise',
                        action='store_true',
                        help='Visualise attention activations during training')

    args = parser.parse_args()

    main(args)
