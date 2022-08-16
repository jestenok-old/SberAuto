import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import requests as r
import json


def main():
    test_df = pd.read_pickle('data/data_with_target.pickle').head(200000)
    pd.DataFrame.to_json()
    print(roc_auc_score(test_df['target'], predict))
    print(confusion_matrix(test_df['target'], predict))


if __name__ == '__main__':
    main()
