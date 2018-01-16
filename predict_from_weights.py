import datetime

import pandas as pd

from preprocessing import load_and_tokenize
from toxic_comments_visualise_attention import build_model
from utils import make_submission

MAX_NUM_WORDS = 30000
MAX_LENGHT = 120

now = datetime.datetime.now()
now = now.strftime('%Y%m%d%H%M')

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')
weight_path = 'weights_base.best.hdf5'
submission_path = './submissions/submission_' + now + '.csv'

class_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
_, _, X_test, word_index, _, _ = load_and_tokenize(class_list,
                                                   MAX_NUM_WORDS,
                                                   MAX_LENGHT)

model, _ = build_model(300, word_index, use_ft=True)
make_submission(model, X_test, class_list, weight_path, submission_path)
