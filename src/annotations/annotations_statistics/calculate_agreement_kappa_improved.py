import json
import os
import re
import string
from collections import Counter

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import data_dir, DISORDER, CHEMICAL_OR_DRUG
from high_recall_matcher_posts_level import words_similarity
from utils import replace_puncs, SequenceTagger

labels_dir = data_dir + r'manual_labeled_v2\doccano'
measuring_kappa_dir = labels_dir + os.sep + "csv_files" + os.sep + "measuring_kappa"

puncs = list(string.punctuation) + ["\n", ' ', '']

def calc_kappa(row, tagger):
    user_5_words_and_tags = row['user_5_bio']['words_and_tags']
    user_6_words_and_tags = row['user_6_bio']['words_and_tags']

    user_5_tokenization_problems = row['user_5_bio']['tokenization_problems']
    user_6_tokenization_problems = row['user_6_bio']['tokenization_problems']

    if len(user_5_tokenization_problems) > 0 or len(user_6_tokenization_problems) > 0:
        user_5_words_and_tags, user_6_words_and_tags = fix_tokenization_problems(row, tagger,
                                                                                 user_5_tokenization_problems,
                                                                                 user_6_tokenization_problems)

    y1 = [x[1] for x in user_5_words_and_tags]
    y2 = [x[1] for x in user_6_words_and_tags]
    kappa = cohen_kappa_score(y1, y2)
    return kappa


def fix_tokenization_problems(row, tagger, user_5_tokenization_problems, user_6_tokenization_problems):
    # print("tokenization problems")
    # print(user_5_tokenization_problems)
    # print(user_6_tokenization_problems)
    bad_terms = user_5_tokenization_problems + user_6_tokenization_problems
    bad_terms_names = [t['term'] for t in bad_terms]
    user_5_labels_fixed = [t for t in row['user_5_labels'] if t['term'] not in bad_terms_names]
    user_6_labels_fixed = [t for t in row['user_6_labels'] if t['term'] not in bad_terms_names]
    if len(user_5_tokenization_problems) > 0:
        assert len(user_5_labels_fixed) < len(row['user_5_labels'])
    if len(user_6_tokenization_problems) > 0:
        assert len(user_6_labels_fixed) < len(row['user_6_labels'])
    row['user_5_labels'] = user_5_labels_fixed
    row['user_6_labels'] = user_6_labels_fixed
    user_5_bio_fixed = tagger.get_bio_tags(row, 'user_5_labels')
    user_6_bio_fixed = tagger.get_bio_tags(row, 'user_6_labels')
    user_5_fixed_tokenization_problems = user_5_bio_fixed['tokenization_problems']
    user_6_fixed_tokenization_problems = user_6_bio_fixed['tokenization_problems']
    assert len(user_5_fixed_tokenization_problems) == 0 and len(user_6_fixed_tokenization_problems) == 0
    user_5_words_and_tags = user_5_bio_fixed['words_and_tags']
    user_6_words_and_tags = user_6_bio_fixed['words_and_tags']
    return user_5_words_and_tags, user_6_words_and_tags


def calculate_kappa_for_community(community):
    df = pd.read_csv(measuring_kappa_dir + os.sep + community + "_both_annotators.csv")

    for c in ['user_5_labels', 'user_6_labels']:
        df[c] = df[c].apply(lambda x: json.loads(x))

    tagger = SequenceTagger(community)

    df['user_5_bio'] = df.apply(lambda row: tagger.get_bio_tags(row, 'user_5_labels'), axis=1)
    df['user_6_bio'] = df.apply(lambda row: tagger.get_bio_tags(row, 'user_6_labels'), axis=1)
    print(f"Finished: {community}, df len: {len(df)}")

    df['kappa'] = df.apply(lambda row: calc_kappa(row, tagger), axis=1)

    print(f'Before filter: len: {len(df)}')
    df = df[~df['kappa'].isna()]
    print(f'After filter: len: {len(df)}')

    kappa_mean = df['kappa'].mean()

    print(f"{community} kappa_mean is: {round(kappa_mean, 3)}")
    print("\n")


def main():
    calculate_kappa_for_community('sclerosis')
    calculate_kappa_for_community('diabetes')
    calculate_kappa_for_community('depression')


if __name__ == '__main__':
    main()
