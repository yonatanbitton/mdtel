import json
import os
import sys
from collections import defaultdict

import pandas as pd

from contextual_relevance.extract_dataset_with_feats.extract_count_features import extract_count_features_for_df
from contextual_relevance.extract_dataset_with_feats.extract_language_model_feats import extract_lm_feats_for_df
from contextual_relevance.extract_dataset_with_feats.extract_relatedness_features import \
    extract_relatedness_feats_for_df
from contextual_relevance.extract_dataset_with_feats.initialize_training_dataset import init_df
from evaluate_contextual_relevance_model import contextual_model_predict_with_trained_model_for_df
from high_recall_matcher import get_high_recall_matches_for_df

module_path = os.path.abspath(os.path.join('..', os.getcwd()))
print(f"In high_recall_matcher, {module_path}")
sys.path.append(module_path)

from config import *

minimal_example_data_path = data_dir + r"minimal_example_post.xlsx"  # First diabetes post
umls_df_data_path = data_dir + r"high_recall_matcher\heb_to_eng_mrconso_disorders_chemicals_kb.csv"
output_dir = data_dir + r"high_recall_matcher\output"
cuiless_dir = data_dir + r"manual_labeled_v2\items_not_in_umls"
acronyms_dir = data_dir + r"manual_labeled_v2\acronyms"


def main():
    # Data - assuming LemLDA pre-processed
    post_df = pd.read_excel(minimal_example_data_path)
    post_row = post_df.iloc[0]
    print(post_row['post_txt'])

    # High recall matcher
    matches_found = get_high_recall_matches_for_df(post_df)
    post_df['matches_found'] = [json.dumps(x) for x in matches_found]
    curr_columns = set(list(post_df.columns))
    print("After high recall matcher, got columns")
    print(curr_columns)
    print(post_df)

    # Contextual relevance model - feature extraction
    ## Preprocess - initialize
    post_df = init_df(post_df)
    print(f"post_df_initialized:")
    init_cols_set = set(list(post_df.columns))
    print("After high recall matcher, got columns")
    print(curr_columns)

    ## 1 - count features
    post_df = extract_count_features_for_df(post_df)
    print(f"Got count & freq features for df")
    count_cols_set = set(post_df.columns)
    print("After count features features, got new columns")
    print(count_cols_set.difference(init_cols_set))

    ## 2 - language model features
    post_df = extract_lm_feats_for_df(post_df)
    lm_cols_set = set(post_df.columns)
    print("After language model features, got new columns")
    print(lm_cols_set.difference(count_cols_set))

    ## 3 - relatedness features - ASSUMING CACHED RELATEDNESS - needs to activate UMLS-Similarity software for new data
    post_df = extract_relatedness_feats_for_df(post_df, DIABETES)  # This post is from the diabetes community
    relatedness_cols_set = set(post_df.columns)
    print("After relatedness features, got new columns")
    print(relatedness_cols_set.difference(lm_cols_set))

    print("All columns, after finishing extracting features for contexual relevance model")
    print(set(post_df.columns))

    # Contextual relevance model - prediction
    preds_for_disorders, preds_for_chemical_and_drugs = contextual_model_predict_with_trained_model_for_df(post_df, DIABETES)
    print("After prediction")
    print('preds_for_disorders')
    print(preds_for_disorders)
    print('preds_for_chemical_and_drugs')
    print(preds_for_chemical_and_drugs)

    print("Done")


def dd2():
    return 0


def dd():
    return defaultdict(dd2)


if __name__ == '__main__':
    main()
