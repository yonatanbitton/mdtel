import argparse
import json
import os
import pickle
import sys
from collections import Counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

from high_recall_matcher_posts_level import words_similarity

module_path = os.path.abspath(os.path.join('..', '..', os.getcwd()))
sys.path.append(module_path)

add_extra_fns = False

def evaluate_community(community):
    print(f"*** community: {community} ***")
    community_df = pd.read_csv(training_dataset_dir + os.sep + community + ".csv")

    print(f"Number of instances: {len(community_df)}, labels distribution:")
    print(community_df['yi'].value_counts())

    if add_extra_fns:
        disorders_number_extra_fns, chemical_or_drugs_extra_fns = get_number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list(community, community_df)
    else:
        disorders_number_extra_fns, chemical_or_drugs_extra_fns = 0, 0

    disorders_community_score = find_best_threshold_for_semantic_type(community, community_df, DISORDER, disorders_number_extra_fns)
    chemicals_community_score = find_best_threshold_for_semantic_type(community, community_df, CHEMICAL_OR_DRUG, chemical_or_drugs_extra_fns)
    return disorders_community_score, chemicals_community_score


def print_best_experiment_data(community, best_experiment_data, semantic_type):
    print(f"{semantic_type} Initial confusion matrix")
    print(best_experiment_data['initial_results']['confusion_matrix'])
    print(f"{semantic_type} Confusion matrix")
    print(best_experiment_data['confusion_matrix'])
    print(f"{semantic_type} False Negatives")
    print(best_experiment_data['initial_results']['cm_examples']['FN'])
    print(f"\n{semantic_type} False Positives")
    print(best_experiment_data['initial_results']['cm_examples']['FP'])
    print(f"{community}-{semantic_type} initial results performace")
    community_score = pd.DataFrame([{k: v for k, v in best_experiment_data['initial_results'].items() if
                                     k in ['f1_score', 'acc', 'precision', 'recall']}], index=[community])
    print(community_score)
    print(f"{community}-{semantic_type} curr results performace")
    community_score = pd.DataFrame(
        [{k: v for k, v in best_experiment_data.items() if k in ['f1_score', 'acc', 'precision', 'recall']}],
        index=[community])
    print(community_score)
    return community_score


def find_best_threshold_for_semantic_type(community, community_df, semantic_type, extra_fns):
    semantic_type_df = community_df[community_df['semantic_type'] == semantic_type]

    selected_feats = ['match_freq', 'pred_3_window', 'pred_6_window', 'relatedness']

    best_joint_score = 0
    best_experiment_data = None
    n_estimators = 50
    test_size = 0.25
    for threshold in [i / 10 for i in list(range(1, 10))]:
        experiment_data, score = get_results_for_threshold(community, semantic_type_df, n_estimators, selected_feats,
                                                           test_size, threshold, extra_fns)

        if score > best_joint_score:
            best_joint_score = score
            best_experiment_data = experiment_data

    community_score = print_best_experiment_data(community, best_experiment_data, semantic_type)

    return community_score


def get_number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list(community, community_df):
    # Adding misses of high recall matcher
    all_high_recall_matches = community_df['cand_match'].values
    labels_df = pd.read_csv(labels_dir + os.sep + community + "_labeled.csv")
    labels_df[FINAL_LABELS_COL] = labels_df[FINAL_LABELS_COL].apply(json.loads)

    labels_df = add_semantic_type_cols(labels_df)

    disorders_number_extra_fns = get_misclassifications_numbers_for_semantic_type(
        all_high_recall_matches, labels_df, DISORDERS_COL)

    chemical_or_drugs_extra_fns = get_misclassifications_numbers_for_semantic_type(
        all_high_recall_matches, labels_df, DISORDERS_COL)

    return disorders_number_extra_fns, chemical_or_drugs_extra_fns


def get_misclassifications_numbers_for_semantic_type(all_high_recall_matches, labels_df, annotated_col):
    all_positive_labeled_terms = []
    for lst in labels_df[annotated_col]:
        all_positive_labeled_terms += lst
    all_positive_labeled_terms = [x for x in all_positive_labeled_terms if
                                  len(x) > 3]  # We don't try to find terms shorter then 3
    missed_items_strict = [x for x in all_positive_labeled_terms if x not in all_high_recall_matches]
    missed_items = difference_with_similarity(all_positive_labeled_terms, all_high_recall_matches)
    print(f"{annotated_col} missed_items_strict {len(missed_items_strict)}")
    print(missed_items_strict)
    print('\n')
    print(f"{annotated_col} missed_items {len(missed_items)}, set length: {len(set(missed_items))}")
    print(set(missed_items))
    print('\n')
    number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list = len(missed_items)
    items_in_high_recall_but_not_labeled = difference_with_similarity(all_high_recall_matches,
                                                                      all_positive_labeled_terms)
    print(f"{annotated_col} items_in_high_recall_but_not_labeled: {len(items_in_high_recall_but_not_labeled)},"
          f" {len(set(items_in_high_recall_but_not_labeled))}")
    print(set(items_in_high_recall_but_not_labeled))
    return number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list


def add_semantic_type_cols(labels_df):
    all_labels_counts = labels_df[FINAL_LABELS_COL].apply(lambda lst: Counter([t['label'] for t in lst])).sum()
    if len(all_labels_counts) > 2:
        raise Exception("Unrecognized label")
    most_common = all_labels_counts.most_common(2)
    disorders_number = most_common[0][0]
    chemicals_or_drug_number = most_common[1][0]
    semantic_type_to_number_dict = {DISORDER: disorders_number, CHEMICAL_OR_DRUG: chemicals_or_drug_number}
    labels_df[DISORDERS_COL] = labels_df[FINAL_LABELS_COL].apply(
        lambda lst: [t['term'] for t in lst if t['label'] == semantic_type_to_number_dict[DISORDER]])
    labels_df[CHEMICAL_OR_DRUGS_COL] = labels_df[FINAL_LABELS_COL].apply(
        lambda lst: [t['term'] for t in lst if t['label'] == semantic_type_to_number_dict[CHEMICAL_OR_DRUG]])
    return labels_df

def difference_with_similarity(all_positive_labeled_terms, all_high_recall_matches):
    missed_items = []
    for x in all_positive_labeled_terms:
        x_in_high_recall_match = False
        for y in all_high_recall_matches:
            if words_similarity(x, y) > SIMILARITY_THRESHOLD:
                x_in_high_recall_match = True
                break
        if not x_in_high_recall_match:
            missed_items.append(x)
    return missed_items


def get_results_for_threshold(community, community_df, n_estimators, selected_feats, test_size, threshold, extra_fns):
    model = RandomForestClassifier(n_estimators=n_estimators)
    X = community_df[selected_feats]
    y = community_df['yi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    model.fit(X_train, y_train)
    y_pred = [x[1] for x in model.predict_proba(X_test)]
    hard_y_pred = [int(x > threshold) for x in y_pred]
    y_test = list(y_test.values)

    cm_examples = get_confusion_matrix_examples(X_test, community_df, hard_y_pred, y_test)

    acc, cm, f1_score_score, precision_val, recall_score_score, roc_auc \
        = get_measures_for_y_test_and_hard_y_pred(hard_y_pred, y_pred, y_test)
    initial_results = {'f1_score': f1_score_score, 'roc_auc': roc_auc, 'acc': acc, 'precision': precision_val,
                       'recall': recall_score_score, 'community': community, 'model': model,
                       'confusion_matrix': cm, 'cm_examples': cm_examples}
    extra_false_negative = int(extra_fns * test_size)

    hard_y_pred = hard_y_pred + [0 for _ in range(extra_false_negative)]
    y_test = y_test + [1 for _ in range(extra_false_negative)]
    acc, cm, f1_score_score, precision_val, recall_score_score, roc_auc \
        = get_measures_for_y_test_and_hard_y_pred(hard_y_pred, y_pred, y_test)

    score = f1_score_score
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    experiment_data = {'f1_score': f1_score_score,'roc_auc': roc_auc, 'acc': acc, 'precision': precision_val,
                       'recall': recall_score_score, 'community': community, 'model': model,
                       'confusion_matrix': cm, 'initial_results': initial_results}
    return experiment_data, score


def get_confusion_matrix_examples(X_test, community_df, hard_y_pred, y_test):
    confusion_matrix_examples = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    for idx, (match_idx, pred, real_ans) in enumerate(zip(list(X_test.index), hard_y_pred, y_test)):
        if pred == real_ans:
            if pred == 1 and real_ans == 1:
                confusion_matrix_examples['TP'].append(community_df.loc[match_idx]['cand_match'])
            elif pred == 0 and real_ans == 0:
                confusion_matrix_examples['TN'].append(community_df.loc[match_idx]['cand_match'])
        else:
            if pred == 1 and real_ans == 0:
                confusion_matrix_examples['FP'].append(community_df.loc[match_idx]['cand_match'])
            elif pred == 0 and real_ans == 1:
                confusion_matrix_examples['FN'].append(community_df.loc[match_idx]['cand_match'])
    return confusion_matrix_examples

def get_measures_for_y_test_and_hard_y_pred(hard_y_pred, y_pred, y_test):
    cm = confusion_matrix(y_test, hard_y_pred)
    f1_score_score = round(f1_score(y_test, hard_y_pred), 2)
    recall_score_score = round(recall_score(y_test, hard_y_pred), 2)
    precision_val = round(precision_score(y_test, hard_y_pred), 2)
    acc = round(accuracy_score(y_test, hard_y_pred), 2)
    if len(y_pred) != len(y_test):
        roc_auc = -1
    else:
        roc_auc = round(roc_auc_score(y_test, y_pred), 2)
    return acc, cm, f1_score_score, precision_val, recall_score_score, roc_auc


def main():
    sclerosis_disorders_community_score, sclerosis_chemicals_community_score = evaluate_community('sclerosis')
    diabetes_disorders_community_score, diabetes_chemicals_community_score = evaluate_community('diabetes')
    depression_disorders_community_score, depression_chemicals_community_score = evaluate_community('depression')

    print("Disorders performance")
    disorders_df = pd.concat([diabetes_disorders_community_score, sclerosis_disorders_community_score, depression_disorders_community_score])
    print(disorders_df)

    print("Chemical or drugs performance")
    chemical_or_drugs_df = pd.concat([diabetes_chemicals_community_score, sclerosis_chemicals_community_score, depression_chemicals_community_score])
    print(chemical_or_drugs_df)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='UMLS Entity Linking Evaluator')
        parser.add_argument('data_dir', help='Path to the input directory.')
        parsed_args = parser.parse_args(sys.argv[1:])
        data_dir = parsed_args.data_dir
        print(f"Got data_dir: {data_dir}")
    else:
        from config import data_dir, FINAL_LABELS_COL, CHEMICAL_OR_DRUG, DISORDER, DISORDERS_COL, CHEMICAL_OR_DRUGS_COL, \
    SIMILARITY_THRESHOLD

    training_dataset_dir = data_dir + r"contextual_relevance\training_dataset_with_labels"
    output_models_dir = data_dir + r"contextual_relevance\output_models"
    labels_dir = data_dir + r'manual_labeled_v2\labels_dataframes'

    main()

