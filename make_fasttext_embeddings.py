import pandas as pd
import fasttext

from preprocessing import text_normalization

USE_EXTRA_COMMENTS = False

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

text_normalization(train)
text_normalization(test)

if USE_EXTRA_COMMENTS:
    extra_comments = pd.read_csv('./data/all_toxicity_comments.csv')
    text_normalization(extra_comments)

    train_comments = train['comment_text'].values
    test_comments = test['comment_text'].values
    toxicity_comments = extra_comments['comment_text'].values

    all_text = list(train_comments) + list(test_comments) + list(toxicity_comments)
else:
    train_comments = train['comment_text'].values
    test_comments = test['comment_text'].values

    all_text = list(train_comments) + list(test_comments)

write_file = open('./data/all_texts.txt', 'w', encoding='utf-8')
for comment in all_text:
    write_file.write('%s' % comment)
write_file.close()
model = fasttext.skipgram('./data/all_texts.txt', 'skipgram-model-300', dim=300)
