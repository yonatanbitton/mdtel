
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
    r1 = pd.merge(initialized_training_dataset, count_features, on=joint_cols)
    r2 = pd.merge(r1, lm_features, on=joint_cols)
    result = pd.merge(r2, relatedness_features, on=joint_cols)
    print(f"shape: {result.shape}, column lengths: {column_lengths}, joint cols: {len(joint_cols)}, new_res: {len(result.columns)}")
    result.to_excel(extracted_training_dataset_dir + os.sep + community + ".xlsx", index=False)
    print(f"merged written to file {extracted_training_dataset_dir + os.sep + community}")

def main():
    for community in ['diabetes', 'sclerosis', 'depression']:
        handle_community(community)

    print("Done")

if __name__ == '__main__':
    main()