import pandas as pd
import fasttext

from feature_engineering import Sanitize
from preprocessing import TextPreprocessor

USE_EXTRA_COMMENTS = True
CBOW = False

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
toxic_comments = pd.read_csv('./data/toxicity_annotated_comments.tsv', sep='\t')
attack_comments = pd.read_csv('./data/attack_annotated_comments.tsv', sep='\t')

TextPreprocessor.text_normalization(train)
TextPreprocessor.text_normalization(test)
toxic_comments = Sanitize(toxic_comments)
attack_comments = Sanitize(attack_comments)


if USE_EXTRA_COMMENTS:
    train_comments = train['comment_text'].values
    test_comments = test['comment_text'].values
    toxic_comments = toxic_comments['comment_text'].values
    attack_comments = attack_comments['comment_text'].values

    all_text = list(train_comments) + list(test_comments) + \
        list(toxic_comments) + list(attack_comments)
else:
    train_comments = train['comment_text'].values
    test_comments = test['comment_text'].values

    all_text = list(train_comments) + list(test_comments)

write_file = open('./data/all_texts.txt', 'w', encoding='utf-8')
for comment in all_text:
    write_file.write('%s' % comment)
write_file.close()

if CBOW:
    model = fasttext.cbow('./data/all_texts.txt', 'cbow-model-300', dim=300)
else:
    model = fasttext.skipgram('./data/all_texts.txt', 'skipgram-model-300', dim=300)
