import dill
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, roc_auc_score


with open('data/sberauto_pipe.pkl', 'rb') as file:
    pipe = dill.load(file)


def main():
    test_df = pd.read_pickle('data/data_with_target.pickle').head(2000)
    predict = pipe['model'].predict(test_df.drop('target', axis=1))

    print(roc_auc_score(test_df['target'], predict))
    print(confusion_matrix(test_df['target'], predict))


if __name__ == '__main__':
    main()
