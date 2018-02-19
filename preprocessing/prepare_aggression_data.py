import pandas as pd

'''
print('----------AGGRESSION----------')
aggression_comments = pd.read_csv('./data/aggression_annotated_comments.tsv',
                                  sep='\t',
                                  index_col=0)
aggression_annotations = pd.read_csv('./data/aggression_annotations.tsv',
                                     sep='\t')

aggression_labels = aggression_annotations.groupby('rev_id')['aggression'].mean() > 0.5
aggression_comments['aggression'] = aggression_labels

aggression_comments['comment'] = aggression_comments['comment'].apply(
    lambda x: x.replace("NEWLINE_TOKEN", " "))
aggression_comments['comment'] = aggression_comments['comment'].apply(
    lambda x: x.replace("TAB_TOKEN", " "))

only_aggressive_comments = aggression_comments.query('aggression')['comment']
print(only_aggressive_comments.head())


print('----------ATTACK----------')
attack_comments = pd.read_csv('./data/attack_annotated_comments.tsv',
                              sep='\t',
                              index_col=0)
attack_annotations = pd.read_csv('./data/attack_annotations.tsv',
                                 sep='\t')
attack_labels = attack_annotations.groupby('rev_id')['attack'].mean() > 0.5
attack_comments['attack'] = attack_labels

attack_comments['comment'] = attack_comments['comment'].apply(
    lambda x: x.replace("NEWLINE_TOKEN", " "))
attack_comments['comment'] = attack_comments['comment'].apply(
    lambda x: x.replace("TAB_TOKEN", " "))

only_attack_comments = attack_comments.query('attack')['comment']
print(only_attack_comments.head())
'''

print('\n----------TOXCICITY----------')
toxicity_comments = pd.read_csv('./data/toxicity_annotated_comments.tsv',
                                sep='\t')
toxicity_annotations = pd.read_csv('./data/toxicity_annotations.tsv',
                                   sep='\t')

toxicity_labels = toxicity_annotations.groupby('rev_id', as_index=False)['toxicity'].mean() > 0.5

toxicity_comments['toxicity'] = toxicity_labels['toxicity']
# toxicity_comments['severe_toxicity'] = severe_toxicity_labels['toxicity']

toxicity_comments['comment'] = toxicity_comments['comment'].apply(
    lambda x: x.replace("NEWLINE_TOKEN", " "))
toxicity_comments['comment'] = toxicity_comments['comment'].apply(
    lambda x: x.replace("TAB_TOKEN", " "))

toxicity_comments.rename(columns={'comment': 'comment_text'}, inplace=True)

all_comments = toxicity_comments['comment_text']
all_comments.to_csv('./data/all_toxicity_comments.csv', index=False, encoding='utf-8')

only_toxic_comments = pd.DataFrame(
    toxicity_comments[toxicity_comments['toxicity']]['comment_text'])

class_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for cat in class_list:
    if cat == 'toxic':
        only_toxic_comments[cat] = 1
    else:
        only_toxic_comments[cat] = 0

only_toxic_comments.to_csv('./data/extra_toxic_comments.csv',
                           sep=',',
                           index=False,
                           encoding='utf-8')
