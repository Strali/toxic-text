import pandas as pd

data = pd.read_csv('./data/train.csv')
comments = data['comment_text']
labels = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

comments.to_csv('./data/train_comments.csv', index=False)
labels.to_csv('./data/train_labels.csv', index=True)
