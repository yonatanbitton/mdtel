import argparse
import json
import os
import sys
from statistics import mode

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

    selected_feats = ['match_freq', 'pred_3_window', 'pred_6_window', 'relatedness'] # 'dep_part', 'gen', 'pos', 'tense'
    categorial_feats = ['dep_part', 'gen', 'pos', 'tense']
    # categorial_feats = ['dep_part', 'pos']

    best_joint_score = 0
    best_experiment_data = None
    n_estimators = 50
    test_size = 0.25
    for threshold in [i / 10 for i in list(range(1, 10))]:
        experiment_data, score = get_results_for_threshold(community, semantic_type_df, n_estimators, selected_feats, categorial_feats,
                                                           test_size, threshold, extra_fns)

        if score > best_joint_score:
            best_joint_score = score
            best_experiment_data = experiment_data

    community_score = print_best_experiment_data(community, best_experiment_data, semantic_type)
    return community_score


def get_number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list(community, community_df):
    # Adding misses of high recall matcher
    labels_df = pd.read_csv(labels_dir + os.sep + community + "_labels.csv")
    labels_df[FINAL_LABELS_COL] = labels_df[FINAL_LABELS_COL].apply(json.loads)

    labels_df = add_semantic_type_cols(labels_df, community)

    chemical_or_drugs_extra_fns = get_misclassifications_numbers_for_semantic_type(community_df, labels_df, CHEMICAL_OR_DRUGS_COL, CHEMICAL_OR_DRUG)
    disorders_number_extra_fns = get_misclassifications_numbers_for_semantic_type(community_df, labels_df, DISORDERS_COL, DISORDER)


    return disorders_number_extra_fns, chemical_or_drugs_extra_fns


def get_misclassifications_numbers_for_semantic_type(community_df, labels_df, annotated_col, semantic_type):

    semantic_type_all_high_recall_matches = community_df[community_df['semantic_type'] == semantic_type]['cand_match'].values

    all_positive_labeled_terms = []
    for lst in labels_df[annotated_col]:
        all_positive_labeled_terms += lst
    all_positive_labeled_terms = [x for x in all_positive_labeled_terms if
                                  len(x) > 3]  # We don't try to find terms shorter then 3
    # items_in_high_recall_but_not_labeled = difference_with_similarity_or_sub_term(semantic_type_all_high_recall_matches,
    #                                                                   all_positive_labeled_terms)
    # print(f"{annotated_col} items_in_high_recall_but_not_labeled: {len(items_in_high_recall_but_not_labeled)},"
    #       f" {len(set(items_in_high_recall_but_not_labeled))}")
    # print(set(items_in_high_recall_but_not_labeled))


    missed_items_strict = [x for x in all_positive_labeled_terms if x not in semantic_type_all_high_recall_matches]
    missed_items = difference_with_similarity(all_positive_labeled_terms, semantic_type_all_high_recall_matches)
    print(f"{annotated_col} missed_items_strict {len(missed_items_strict)}")
    print(f"{annotated_col} missed_items {len(missed_items)}, set length: {len(set(missed_items))}")
    number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list = len(missed_items)
    print(missed_items)
    # number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list = 0 # TODO - update
    return number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list


def add_semantic_type_cols(labels_df, community):
    comm_to_heb = {'diabetes': 'סוכרת', 'sclerosis': 'טרשת נפוצה', 'depression': 'דיכאון'}
    labels_df[FINAL_LABELS_COL].apply(lambda lst: [m for m in lst if m['term'] == comm_to_heb[community]])
    labels_df_community_name = labels_df[FINAL_LABELS_COL].apply(lambda lst: [m for m in lst if m['term'] == comm_to_heb[community]])
    community_name_user_values = [lst[0]['label'] for lst in labels_df_community_name.values if lst != []]
    print(f"comm label names")
    disorders_number = mode(community_name_user_values)
    different_num_series = labels_df[FINAL_LABELS_COL].apply(lambda lst: [m for m in lst if m['label'] != disorders_number])
    chemicals_or_drug_number = mode([lst[0]['label'] for lst in different_num_series.values if lst != []])
    print(f"disorders_number: {disorders_number}, community_name_user_values: {community_name_user_values[:5]}, chemicals_or_drug_number: {chemicals_or_drug_number}")
    semantic_type_to_number_dict = {DISORDER: disorders_number, CHEMICAL_OR_DRUG: chemicals_or_drug_number}
    labels_df[DISORDERS_COL] = labels_df[FINAL_LABELS_COL].apply(
        lambda lst: [t['term'] for t in lst if t['label'] == semantic_type_to_number_dict[DISORDER]])
    labels_df[CHEMICAL_OR_DRUGS_COL] = labels_df[FINAL_LABELS_COL].apply(
        lambda lst: [t['term'] for t in lst if t['label'] == semantic_type_to_number_dict[CHEMICAL_OR_DRUG]])
    return labels_df

def difference_with_similarity(l1, l2):
    missed_items = []
    for x in l1:
        x_in_l2 = False
        for y in l2:
            if words_similarity(x, y) > SIMILARITY_THRESHOLD:
                x_in_l2 = True
                break
        if not x_in_l2:
            missed_items.append(x)
    return missed_items


def subterm_is_inside(x, y):
    x_len = len(x.split(" "))
    y_len = len(y.split(" "))
    if y_len > x_len:
        y_same_length_as_x = " ".join(y.split(" ")[:x_len])
        if words_similarity(x, y_same_length_as_x) > SIMILARITY_THRESHOLD:
            return True
    return False


def difference_with_similarity_or_sub_term(semantic_type_all_high_recall_matches, all_positive_labeled_terms):
    missed_items = []
    for x in semantic_type_all_high_recall_matches:
        if x == 'סטרואיד' in x:
            print("HERE!")
        x_in_all_positive_labeled_terms = x in all_positive_labeled_terms
        if not x_in_all_positive_labeled_terms:
            for y in all_positive_labeled_terms:
                if words_similarity(x, y) > SIMILARITY_THRESHOLD or subterm_is_inside(x, y):
                    x_in_all_positive_labeled_terms = True
                    break
        if not x_in_all_positive_labeled_terms:
            missed_items.append(x)
    return missed_items



def get_results_for_threshold(community, community_df, n_estimators, selected_feats, categorial_feats, test_size, threshold, extra_fns):
    model = RandomForestClassifier(n_estimators=n_estimators)
    X = community_df
    y = community_df['yi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train_nums = X_train[selected_feats]
    X_train_dummies = X_train[categorial_feats]
    X_test_nums = X_test[selected_feats]
    X_test_dummies = X_test[categorial_feats]
    X_train_dummies_mat = pd.get_dummies(X_train_dummies)
    X_test_dummies_mat = pd.get_dummies(X_test_dummies)
    X_test_dummies_mat = pd.DataFrame(X_test_dummies_mat, columns=X_train_dummies_mat.columns)
    X_train_dummies_mat.fillna(-1, inplace=True)
    X_test_dummies_mat.fillna(-1, inplace=True)
    X_train = pd.concat([X_train_nums, X_train_dummies_mat], axis=1, sort=False)
    X_test = pd.concat([X_test_nums, X_test_dummies_mat], axis=1, sort=False)
    # print(f"shapes: {X_train_nums.shape, X_train_dummies_mat.shape, X_train.shape}")
    # print(f"shapes: {X_test_nums.shape, X_test_dummies_mat.shape, X_test.shape}")
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
    # model = RandomForestClassifier(n_estimators=n_estimators)
    # model.fit(X, y)
    experiment_data = {'f1_score': f1_score_score,'roc_auc': roc_auc, 'acc': acc, 'precision': precision_val,
                       'recall': recall_score_score, 'community': community, 'model': model,
                       'confusion_matrix': cm, 'initial_results': initial_results}
    return experiment_data, score


def get_confusion_matrix_examples(X_test, community_df, hard_y_pred, y_test):
    confusion_matrix_examples = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    for idx, (match_idx, pred, real_ans) in enumerate(zip(list(X_test.index), hard_y_pred, y_test)):
        item_to_add = community_df.loc[match_idx]['cand_match']
        if pred == real_ans:
            if pred == 1 and real_ans == 1:
                confusion_matrix_examples['TP'].append(item_to_add)
            elif pred == 0 and real_ans == 0:
                confusion_matrix_examples['TN'].append(item_to_add)
        else:
            if pred == 1 and real_ans == 0:
                confusion_matrix_examples['FP'].append(item_to_add)
            elif pred == 0 and real_ans == 1:
                confusion_matrix_examples['FN'].append(item_to_add)
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
    avg = disorders_df.mean().apply(lambda x: round(x, 2))
    disorders_df.loc['average'] = avg
    print(disorders_df)

    print("Chemical or drugs performance")
    chemical_or_drugs_df = pd.concat([diabetes_chemicals_community_score, sclerosis_chemicals_community_score, depression_chemicals_community_score])
    avg = chemical_or_drugs_df.mean().apply(lambda x: round(x, 2))
    chemical_or_drugs_df.loc['average'] = avg
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

    training_dataset_dir = data_dir + r"contextual_relevance\extracted_training_dataset"
    output_models_dir = data_dir + r"contextual_relevance\output_models"
    # labels_dir = data_dir + r'manual_labeled_v2\labels_dataframes'
    labels_dir = data_dir + r'manual_labeled_v2\doccano\merged_output'

    main()

