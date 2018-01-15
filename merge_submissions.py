import glob
import pandas as pd


if __name__ == '__main__':
    sample_submission = pd.read_csv('./data/sample_submission.csv')
    class_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    sample_submission[class_list] = [0]*len(class_list)

    submission_files = glob.glob('./submissions/*.csv')

    WEIGH_SUBMISSIONS = False
    total_weight = 0

    for i, sub in enumerate(submission_files):
        next_sub = pd.read_csv(sub)
        sample_submission[class_list] += next_sub[class_list]

    sample_submission[class_list] /= len(submission_files)
    sample_submission.to_csv('submissions/average_submissions.csv',
                             index=False)
