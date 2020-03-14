import argparse
import json
import os
import pickle
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

from debug.add_labels_old_to_delete import remove_bad_char_and_lower
from high_recall_matcher_posts_level import words_similarity

module_path = os.path.abspath(os.path.join('..', '..', os.getcwd()))
sys.path.append(module_path)

SIMILARITY_THRESHOLD = 0.88

def evaluate_community(community):
    print(f"*** community: {community} ***")
    # community_df = pd.read_csv(training_dataset_dir + os.sep + community + "_debug.csv")
    community_df = pd.read_csv(training_dataset_dir + os.sep + community + ".csv")

    print(f"Number of instances: {len(community_df)}, labels distribution: ")
    print(community_df['yi'].value_counts())

    number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list = get_number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list(
        community, community_df)

    selected_feats = ['match_freq', 'pred_3_window', 'pred_6_window', 'relatedness']

    best_joint_score = 0
    best_experiment_data = None
    n_estimators = 50
    test_size = 0.25
    for threshold in [i/10 for i in list(range(1, 10))]:
        experiment_data, score = get_results_for_threshold(community, community_df, n_estimators, selected_feats,
                                                           test_size, threshold, number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list)

        if score > best_joint_score:
            best_joint_score = score
            best_experiment_data = experiment_data

    print("Initial confusion matrix")
    print(best_experiment_data['initial_results']['confusion_matrix'])
    print("Confusion matrix")
    print(best_experiment_data['confusion_matrix'])

    print("False Negatives")
    print(best_experiment_data['initial_results']['cm_examples']['FN'])
    print("\nFalse Positives")
    print(best_experiment_data['initial_results']['cm_examples']['FP'])

    print(f"{community} initial results performace")
    community_score = pd.DataFrame([{k:v for k,v in best_experiment_data['initial_results'].items() if k in ['f1_score', 'acc', 'precision' ,'recall']}], index=[community])
    print(community_score)

    print(f"{community} curr results performace")
    community_score = pd.DataFrame([{k:v for k,v in best_experiment_data.items() if k in ['f1_score', 'acc', 'precision' ,'recall']}], index=[community])
    print(community_score)

    return best_experiment_data


def get_number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list(community, community_df):
    # Adding misses of high recall matcher
    all_high_recall_matches = community_df['cand_match'].values
    labels_df = pd.read_csv(labels_dir + os.sep + community + "_labeled.csv")
    labels_df[FINAL_LABELS_COL] = labels_df[FINAL_LABELS_COL].apply(json.loads)
    # labels_df[FINAL_LABELS_COL] = labels_df[FINAL_LABELS_COL].apply(lambda x: x.split(",") if str(x) != 'nan' else [])
    labels_df[FINAL_LABELS_COL] = labels_df[FINAL_LABELS_COL].apply(lambda lst: [remove_bad_char_and_lower(x.strip()) for x in lst])

    all_positive_labeled_terms = []
    for lst in labels_df[FINAL_LABELS_COL]:
        all_positive_labeled_terms += lst
    all_positive_labeled_terms = [x for x in all_positive_labeled_terms if len(x) > 3]

    missed_items_strict = [x for x in all_positive_labeled_terms if x not in all_high_recall_matches]
    missed_items = difference_with_similarity(all_positive_labeled_terms, all_high_recall_matches)

    print(f"missed_items_strict {len(missed_items_strict)}")
    print(missed_items_strict)
    print('\n')
    print(f"missed_items {len(missed_items)}")
    print(missed_items)
    print('\n')
    number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list = len(missed_items)

    # items_in_high_recall_but_not_labeled = [x for x in all_high_recall_matches if x not in all_positive_labeled_terms]
    # print(f"items_in_high_recall_but_not_labeled (check noisy label): {items_in_high_recall_but_not_labeled}")
    # print(items_in_high_recall_but_not_labeled)
    items_in_high_recall_but_not_labeled = difference_with_similarity(all_high_recall_matches, all_positive_labeled_terms)
    print(f"items_in_high_recall_but_not_labeled: {len(items_in_high_recall_but_not_labeled)}, {len(set(items_in_high_recall_but_not_labeled))}")
    print(set(items_in_high_recall_but_not_labeled))

    return number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list


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


def get_results_for_threshold(community, community_df, n_estimators, selected_feats, test_size, threshold, number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list):
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
    extra_false_negative = int(number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list * test_size)

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
    sclerosis_model_data = evaluate_community('sclerosis')
    diabetes_model_data = evaluate_community('diabetes')
    depression_model_data = evaluate_community('depression')

    res_df = pd.DataFrame([diabetes_model_data, sclerosis_model_data, depression_model_data], index=['diabetes', 'sclerosis', 'depression'], columns=['f1_score', 'recall', 'acc', 'precision'])
    print(res_df)

    trained_models = {'diabetes': diabetes_model_data, 'sclerosis': sclerosis_model_data,
                      'depression': depression_model_data}

    with open(output_models_dir + os.sep + "trained_models.pickle", 'wb') as f:
        pickle.dump(trained_models, f)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='UMLS Entity Linking Evaluator')
        parser.add_argument('data_dir', help='Path to the input directory.')
        parsed_args = parser.parse_args(sys.argv[1:])
        data_dir = parsed_args.data_dir
        print(f"Got data_dir: {data_dir}")
    else:
        from config import data_dir, FINAL_LABELS_COL

    training_dataset_dir = data_dir + r"contextual_relevance\training_dataset_with_labels"
    output_models_dir = data_dir + r"contextual_relevance\output_models"
    labels_dir = data_dir + r'manual_labeled_v2\labels_dataframes'

    main()

