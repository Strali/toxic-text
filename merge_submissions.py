import datetime
import glob

import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':
    now = datetime.datetime.now()
    now = now.strftime('%Y%m%d')
    SUBMISSION_SAVE_PATH = './submissions/average_submission_' + now + '.csv'
    WEIGH_SUBMISSIONS = False
    GEOMETRIC_MEAN = True

    sample_submission = pd.read_csv('./data/sample_submission.csv')
    class_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    submission_files = glob.glob('./submissions/*.csv')

    if GEOMETRIC_MEAN:
        sample_submission[class_list] = [1] * len(class_list)
    else:
        sample_submission[class_list] = [0] * len(class_list)
        total_weight = 0

    for i, sub in tqdm(enumerate(submission_files)):
        next_sub = pd.read_csv(sub)
        if GEOMETRIC_MEAN:
            sample_submission[class_list] *= next_sub[class_list]
        else:
            weight = 1
            if WEIGH_SUBMISSIONS and 'average' in sub:
                weight = 2
                total_weight += 1
            sample_submission[class_list] += weight*next_sub[class_list]

    if GEOMETRIC_MEAN:
        sample_submission[class_list] **= (1.0 / len(submission_files))
    else:
        sample_submission[class_list] /= (len(submission_files) + total_weight)
    sample_submission.to_csv(SUBMISSION_SAVE_PATH,
                             index=False)
