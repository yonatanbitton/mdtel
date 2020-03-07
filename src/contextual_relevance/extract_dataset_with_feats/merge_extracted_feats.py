
import pandas as pd
import os
from config import data_dir

initialized_training_dataset_dir = data_dir + r"contextual_relevance\initialized_training_dataset"
count_features_path = data_dir + r"contextual_relevance\count_features"
lm_features_path = data_dir + r"contextual_relevance\language_models\output"
relatedness_features_path = data_dir + r"contextual_relevance\relatedness\output"

extracted_training_dataset_dir = data_dir + r"contextual_relevance\extracted_training_dataset"


def handle_community(community):
    initialized_training_dataset = pd.read_excel(initialized_training_dataset_dir + os.sep + community + ".xlsx")
    count_features = pd.read_excel(count_features_path + os.sep + community + "_output.xlsx")
    lm_features = pd.read_excel(lm_features_path + os.sep + community + "_output.xlsx")
    relatedness_features = pd.read_excel(relatedness_features_path + os.sep + community + "_output.xlsx")

    column_lengths = len(initialized_training_dataset.columns), len(count_features.columns), len(lm_features.columns), len(relatedness_features.columns)

    joint_cols = ['match', 'match_10_window', 'match_3_window', 'match_6_window',
                      'match_idx', 'row_idx', 'tokenized_txt']
    count_cols = ['match_count', 'match_freq']
    lm_cols = ['pred_10_window', 'pred_6_window', 'pred_3_window', 'pred_2_window']
    relatedness_cols = ['relatedness']
    all_final_cols = joint_cols + count_cols + lm_cols + relatedness_cols

    print(f"Len before: {len(initialized_training_dataset), len(count_features), len(lm_features), len(relatedness_features)}")

    all_rows = []
    for idx in range(len(initialized_training_dataset)):
        r1 = initialized_training_dataset.iloc[idx]
        r2 = count_features.iloc[idx]
        r3 = lm_features.iloc[idx]
        r4 = relatedness_features.iloc[idx]
        merged_d = {**dict(r1), **dict(r2), **dict(r3), **dict(r4)}
        assert r1['match'] == r2['match'] == r3['match'] == r4['match'] and \
               r1['match_idx'] == r2['match_idx'] == r3['match_idx'] == r4['match_idx'] and \
               r1['row_idx'] == r2['row_idx'] == r3['row_idx'] == r4['row_idx']
        all_rows.append(merged_d)

    result = pd.DataFrame(all_rows, columns=all_final_cols)
    print(f"shapes before: {result.shape}, column lengths: {column_lengths}, joint cols: {len(joint_cols)}, new_res: {len(result.columns)}")
    result.to_excel(extracted_training_dataset_dir + os.sep + community + ".xlsx", index=False)
    print(f"merged written to file {extracted_training_dataset_dir + os.sep + community}")

def main():
    handle_community('diabetes')
    # handle_community('sclerosis')
    # handle_community('depression')

    print("Done")

if __name__ == '__main__':
    main()