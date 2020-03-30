import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from high_recall_matcher_posts_level import words_similarity
from seqeval.metrics import accuracy_score as seq_eval_acc
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score as seq_eval_f1
from seqeval.metrics import performance_measure
from seqeval.metrics import precision_score as seq_eval_precision
from seqeval.metrics import recall_score as seq_eval_recall
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, precision_score
from sklearn_crfsuite import metrics

from utils import SequenceTagger

module_path = os.path.abspath(os.path.join('..', '..', os.getcwd()))
sys.path.append(module_path)

from collections import namedtuple, Counter, defaultdict

Entity = namedtuple("Entity", "e_type start_offset end_offset")
DebugEntity = namedtuple("T", "s_t i s e")

matches_per_community = {'diabetes': {'FN': Counter(), 'FP': Counter(), 'TP': Counter()},
                          'sclerosis': {'FN': Counter(), 'FP': Counter(), 'TP': Counter()},
                          'depression': {'FN': Counter(), 'FP': Counter(), 'TP': Counter()}}

filenames_per_community = {'diabetes': {'FN': defaultdict(list), 'FP': defaultdict(list), 'TP': defaultdict(list)},
                          'sclerosis': {'FN': defaultdict(list), 'FP': defaultdict(list), 'TP': defaultdict(list)},
                          'depression': {'FN': defaultdict(list), 'FP': defaultdict(list), 'TP': defaultdict(list)}
                           }

n_most_common = 10


def evaluate_community(community):
    print(f"*** community: {community} ***")
    train_df, test_df, labels_df = get_data(community)

    disorders_experiment_data = train_semantic_type(community, train_df, test_df, DISORDER)
    chemicals_experiment_data = train_semantic_type(community, train_df, test_df, CHEMICAL_OR_DRUG)

    test_labels_df = get_test_df_with_preds_and_annotations(community, test_df, labels_df, chemicals_experiment_data, disorders_experiment_data)

    token_level_seq_eval_df, token_level_crf_df, token_level_io_crf_df, token_level_io_seq_eval_df, TN = token_evaluation(community, test_labels_df)
    entity_level_eval_df = entity_level_evaluation(community, TN)

    print_errors(community, 'FN')
    print_errors(community, 'FP')

    res_d = {'entity_level_eval_df': entity_level_eval_df,
             'token_level_seq_eval_df': token_level_seq_eval_df,
             'token_level_crf_df': token_level_crf_df,
             'token_level_io_seq_eval_df': token_level_io_seq_eval_df,
             'token_level_io_crf_df': token_level_io_crf_df
             }

    return res_d


def print_errors(community, error_type):
    most_common_error = matches_per_community[community][error_type].most_common(n_most_common)
    print(f"most_common {error_type}s: {community}, number: {len(matches_per_community[community][error_type])}")
    for p in most_common_error:
        print(p, filenames_per_community[community][error_type][p[0]])


def get_test_df_with_preds_and_annotations(community, test_df, labels_df, chemicals_experiment_data, disorders_experiment_data):
    chemicals_seq_data, disorders_seq_data, grouped = get_data_from_training_and_testing(chemicals_experiment_data, disorders_experiment_data, test_df)
    test_labels_df = add_predictions_to_test_df(community, grouped, labels_df, test_df, chemicals_seq_data, disorders_seq_data)
    return test_labels_df


def entity_level_evaluation(community, TN):
    FP = sum(matches_per_community[community]['FP'].values())
    TP = sum(matches_per_community[community]['TP'].values())
    FN = sum(matches_per_community[community]['FN'].values())
    eval_d = get_metrics_based_on_fn_fp_tp(FN, FP, TP, TN)
    entity_level_eval_df = pd.DataFrame([eval_d], index=[community])
    return entity_level_eval_df


def get_metrics_based_on_fn_fp_tp(FN, FP, TP, TN):
    P = TP + FP
    recall = round((TP / P), 2)
    precision = round((TP / (TP + FP)), 2)
    f1_score_val = round((2 * TP / (2 * TP + FP + FN)), 2)
    accuracy = round((TP + TN) / (TP + TN + FP + FN), 2)
    cm = np.array([[TN, FP], [FN, TP]])
    scores = {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1_score': f1_score_val, 'cm': cm}
    return scores


def get_data(community):
    train_community_df = pd.read_csv(training_dataset_dir + os.sep + 'train' + os.sep + community + ".csv")
    test_community_df = pd.read_csv(training_dataset_dir + os.sep + 'test' + os.sep + community + ".csv")
    labels_df = pd.read_csv(labels_dir + os.sep + community + "_labels.csv")
    labels_df[FINAL_LABELS_COL] = labels_df[FINAL_LABELS_COL].apply(lambda x: json.loads(x))

    print(f"Number of instances: train: {len(train_community_df)}, test: {len(test_community_df)}:")
    print(f"Train & test label distribution:")
    print(train_community_df['yi'].value_counts())
    print(test_community_df['yi'].value_counts())

    return train_community_df, test_community_df, labels_df


def assert_terms_in_place(row):
    text = row['text']
    annotations = row[FINAL_LABELS_COL]
    predictions = row['prediction_labels']
    local_sim_thresh = 0.72

    for ann in annotations:
        assert words_similarity(text[ann['start_offset']:ann['end_offset']], ann['term']) > local_sim_thresh

    not_inplace_terms = []
    for p in predictions:
        sim = words_similarity(text[p['start_offset']:p['end_offset']], p['term'])
        if sim < local_sim_thresh:
            not_inplace_terms.append(p)
    if len(not_inplace_terms):
        # print(f"not_inplace_terms: {not_inplace_terms}")
        # print(f"anns: {[t['term'] for t in annotations]}")
        predictions = [p for p in predictions if p not in not_inplace_terms]

    return predictions


def get_entity_named_tuple(lst):
    all_ents = []
    for t in lst:
        label = t['label']
        ent = Entity(label, t['start_offset'], t['end_offset'])
        all_ents.append(ent)
    return all_ents


def bio_to_io(lst):
    return [t.split("-")[1] if t != 'O' else 'O' for t in lst ]

def token_evaluation(community, test_labels_df):

    y_pred, y_true = get_bio(community, test_labels_df)

    labels = ['B-D', 'I-D', 'B-C', 'I-C']
    token_level_crf_df, token_level_seq_eval_df = get_token_eval_for_y_true_pred(community, y_pred, y_true, labels)

    y_true_i_o = [bio_to_io(lst) for lst in y_true]
    y_pred_i_o = [bio_to_io(lst) for lst in y_pred]
    labels = ['D', 'C']
    token_level_io_crf_df, token_level_io_seq_eval_df = get_token_eval_for_y_true_pred(community, y_true_i_o, y_pred_i_o, labels)

    TN = get_tn(y_pred, y_true)

    return token_level_seq_eval_df, token_level_crf_df, token_level_io_crf_df, token_level_io_seq_eval_df, TN


def get_token_eval_for_y_true_pred(community, y_pred, y_true, labels):
    seq_eval_scores = get_seqeval_metrics(y_pred, y_true)
    crf_scores = get_crf_metrics(y_pred, y_true, labels)
    token_level_seq_eval_df = pd.DataFrame([seq_eval_scores], index=[community])
    token_level_crf_df = pd.DataFrame([crf_scores], index=[community])
    return token_level_crf_df, token_level_seq_eval_df


def get_tn(y_pred, y_true):
    y_true_flat = [item for sublist in y_true for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]
    TN = sum((y_t == y_p == 'O') for y_t, y_p in zip(y_true_flat, y_pred_flat))
    return TN


def get_bio(community, test_labels_df):
    annotation_tagger = SequenceTagger(community)
    pred_tagger = SequenceTagger(community, predictor=True)
    all_y_true = []
    all_y_pred = []
    for idx, row in test_labels_df.iterrows():
        post_y_true, post_y_pred = get_bio_for_row(annotation_tagger, pred_tagger, row)
        all_y_true.append(post_y_true)
        all_y_pred.append(post_y_pred)
    y_true = all_y_true
    y_pred = all_y_pred
    return y_pred, y_true


def get_bio_for_row(annotation_tagger, pred_tagger, row):
    global matches_per_community
    annotation_bio_tags = annotation_tagger.get_bio_tags(row, FINAL_LABELS_COL)
    prediction_bio_tags = pred_tagger.get_bio_tags(row, 'prediction_labels')
    post_y_true = [x[1] for x in annotation_bio_tags['words_and_tags']]
    post_y_pred = [x[1] for x in prediction_bio_tags['words_and_tags']]
    words = [x[0] for x in annotation_bio_tags['words_and_tags']]
    word_true_pred = list(zip(words, post_y_true, post_y_pred))

    assert len(post_y_true) == len(post_y_pred)
    return post_y_true, post_y_pred

def get_seqeval_metrics(y_pred, y_true):
    seq_eval_token_f1_score = round(seq_eval_f1(y_true, y_pred), 2)
    seq_eval_token_acc_score = round(seq_eval_acc(y_true, y_pred), 2)
    seq_eval_token_recall_score = round(seq_eval_recall(y_true, y_pred), 2)
    seq_eval_token_precision_score = round(seq_eval_precision(y_true, y_pred), 2)
    report = classification_report(y_true, y_pred, digits=2)
    cm_dict = performance_measure(y_true, y_pred)
    cm = np.array([[cm_dict['TN'], cm_dict['FP']], [cm_dict['FN'], cm_dict['TP']]])
    res_d = {'accuracy': seq_eval_token_acc_score, 'recall': seq_eval_token_recall_score, 'f1_score': seq_eval_token_f1_score,
             'precision': seq_eval_token_precision_score, 'cm': cm, 'report': report}
    return res_d


def get_crf_metrics(y_pred, y_true, labels):
    token_acc_score = round(metrics.flat_accuracy_score(y_true, y_pred), 2)
    token_recall_score = round(metrics.flat_recall_score(y_true, y_pred, average='weighted', labels=labels), 2)
    token_f1_score = round(metrics.flat_f1_score(y_true, y_pred, average='weighted', labels=labels), 2)
    token_precision_score = round(metrics.flat_precision_score(y_true, y_pred, average='weighted', labels=labels), 2)
    report = metrics.flat_classification_report(y_true, y_pred, labels=labels)
    cm_dict = performance_measure(y_true, y_pred)
    cm = np.array([[cm_dict['TN'], cm_dict['FP']], [cm_dict['FN'], cm_dict['TP']]])
    res_d = {'accuracy': token_acc_score, 'recall': token_recall_score, 'f1_score': token_f1_score,
             'precision': token_precision_score, 'cm': cm, 'report': report}
    return res_d


def get_items_in_left_lst_but_not_in_right_lst(l1, l2):
    in_l1_but_not_in_l2 = []
    for ann in l1:
        found_pred = False
        for pred in l2:
            if words_similarity(ann['term'], pred['term']) > 0.8 and abs(
                    ann['start_offset'] - pred['start_offset']) <= 2:
                found_pred = True
                break
        if not found_pred:
            in_l1_but_not_in_l2.append(ann)

    return in_l1_but_not_in_l2

def get_items_in_both_lsts(l1, l2):
    in_both_lsts = []
    for ann in l1:
        found_pred = False
        for pred in l2:
            if words_similarity(ann['term'], pred['term']) > 0.8 and abs(
                    ann['start_offset'] - pred['start_offset']) <= 2:
                if ann['label'] == pred['label']:
                    found_pred = True
                    break
        if found_pred:
            in_both_lsts.append(ann)

    return in_both_lsts


def add_predictions_to_test_df(community, grouped, labels_df, test_df, chemicals_seq_data, disorders_seq_data):
    labels_per_post = {}
    for idx, post_df in grouped:
        post_file_name = post_df['file_name'].iloc[0]
        post_disorders_data = disorders_seq_data[disorders_seq_data['file_name'] == post_file_name]
        post_disorders_data = post_disorders_data[post_disorders_data['hard_y_pred'] == 1]
        post_chemicals_data = chemicals_seq_data[chemicals_seq_data['file_name'] == post_file_name]
        post_chemicals_data = post_chemicals_data[post_chemicals_data['hard_y_pred'] == 1]
        disorders_labels = get_labels_for_semantic_type(post_disorders_data)
        chemical_labels = get_labels_for_semantic_type(post_chemicals_data)
        joint_labels = disorders_labels + chemical_labels
        if post_file_name in labels_per_post:
            print(f"Problem. Double. {post_file_name} - {community}")
            raise Exception("Double id")
        labels_per_post[post_file_name] = joint_labels
    test_labels_df = labels_df[labels_df['file_name'].isin(set(test_df['file_name']))]
    test_labels_df['prediction_labels'] = test_labels_df['file_name'].apply(
        lambda post_file_name: labels_per_post[post_file_name])
    test_labels_df['prediction_labels'] = test_labels_df.apply(lambda row: assert_terms_in_place(row), axis=1)
    test_labels_df['prediction_named_tuple'] = test_labels_df['prediction_labels'].apply(lambda lst: get_entity_named_tuple(lst))
    test_labels_df['annotation_named_tuple'] = test_labels_df['merged_inner_and_outer'].apply(lambda lst: get_entity_named_tuple(lst))
    test_labels_df['predicted_and_annotated'] = test_labels_df.apply(lambda row: get_items_in_both_lsts(row[FINAL_LABELS_COL], row['prediction_labels']), axis=1)
    test_labels_df['predicted_not_annotated'] = test_labels_df.apply(lambda row: get_items_in_left_lst_but_not_in_right_lst(row['prediction_labels'], row[FINAL_LABELS_COL]), axis=1)
    test_labels_df['annotated_not_predicted'] = test_labels_df.apply(lambda row: get_items_in_left_lst_but_not_in_right_lst(row[FINAL_LABELS_COL], row['prediction_labels']), axis=1)

    add_types_to_counter(community, 'FN', 'annotated_not_predicted', test_labels_df)
    add_types_to_counter(community, 'FP', 'predicted_not_annotated', test_labels_df)
    add_types_to_counter(community, 'TP', 'predicted_and_annotated', test_labels_df)

    return test_labels_df


def add_types_to_counter(community, dict_key, row_type, test_labels_df):
    for row_idx, row in test_labels_df.iterrows():
        annotated_not_predicted = row[row_type]
        file_name = row['file_name']
        for t in annotated_not_predicted:
            filenames_per_community[community][dict_key][t['term']].append(DebugEntity(file_name, t['label'], t['start_offset'], t['end_offset']))
        matches_per_community[community][dict_key] += Counter([m['term'] for m in annotated_not_predicted])


def get_data_from_training_and_testing(chemicals_experiment_data, disorders_experiment_data, test_df):
    disorders_seq_data = disorders_experiment_data['X_test_seq_data']
    chemicals_seq_data = chemicals_experiment_data['X_test_seq_data']
    grouped = test_df.groupby('file_name')
    return chemicals_seq_data, disorders_seq_data, grouped


def get_labels_for_semantic_type(post_semantic_type_data):
    post_labels = []
    for idx, row in post_semantic_type_data.iterrows():
        d = {'term': row['cand_match'], 'umls_match': row['umls_match'],
             'start_offset': row['curr_occurence_offset'],
             'end_offset': row['curr_occurence_offset'] + len(row['cand_match']),
             'label': row['semantic_type']
             }
        post_labels.append(d)
    return post_labels


def train_semantic_type(community, train_df, test_df, semantic_type):
    train_semantic_type_df = train_df[train_df['semantic_type'] == semantic_type]
    test_semantic_type_df = test_df[test_df['semantic_type'] == semantic_type]
    selected_feats = ['match_count', 'match_freq', 'pred_3_window', 'pred_6_window', 'pred_10_window', 'relatedness']
    # selected_feats = ['match_freq', 'pred_3_window', 'pred_6_window', 'relatedness']

    n_estimators = 50

    model = RandomForestClassifier(n_estimators=n_estimators)

    X_train = train_semantic_type_df
    y_train = X_train['yi']
    X_test = test_semantic_type_df
    y_test = X_test['yi']
    X_train_matrix = X_train[selected_feats]
    X_test_matrix = X_test[selected_feats]
    model.fit(X_train_matrix, y_train)
    y_pred = [x[1] for x in model.predict_proba(X_test_matrix)]

    best_experiment_data = find_best_threshold(X_test, community, y_pred, y_test)

    return best_experiment_data


def find_best_threshold(X_test, community, y_pred, y_test):
    best_joint_score = 0
    best_experiment_data = None

    for threshold in [i / 10 for i in list(range(1, 10))]:
        experiment_data, score = get_results_for_threshold(threshold, community, y_test, y_pred)

        if score > best_joint_score:
            best_joint_score = score
            best_experiment_data = experiment_data
    X_test['hard_y_pred'] = best_experiment_data['hard_y_pred']
    X_test_seq_data = X_test[
        ['cand_match', 'umls_match', 'hard_y_pred', 'yi', 'file_name', 'match_eng', 'semantic_type',
         'curr_occurence_offset', 'match_6_window']]
    best_experiment_data['X_test_seq_data'] = X_test_seq_data
    return best_experiment_data


def get_results_for_threshold(threshold, community, y_test, y_pred):
    hard_y_pred = [int(x > threshold) for x in y_pred]
    y_test = list(y_test.values)

    acc, cm, f1_score_score, precision_val, recall_score_score, roc_auc \
        = get_measures_for_y_test_and_hard_y_pred(hard_y_pred, y_pred, y_test)
    experiment_data = {'f1_score': f1_score_score, 'roc_auc': roc_auc, 'accuracy': acc, 'precision': precision_val,
                       'recall': recall_score_score, 'community': community, 'confusion_matrix': cm,
                       'hard_y_pred': hard_y_pred}

    score = f1_score_score
    return experiment_data, score


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
    depression_results = evaluate_community('depression')
    diabetes_results = evaluate_community('diabetes')
    sclerosis_results = evaluate_community('sclerosis')

    titles_for_keys = {'entity_level_eval_df': 'Entity level eval',
                       'token_level_seq_eval_df': 'Token level seqeval performance',
                       'token_level_crf_df': 'Token level crfsuite performance',
                       'token_level_io_crf_df': 'Token level IO tags seqeval performance',
                       'token_level_io_seq_eval_df': 'Token level IO tags crfsuite performance',
                       }

    for k, title in titles_for_keys.items():
        print_res_df(diabetes_results[k], sclerosis_results[k], depression_results[k], title=title)

    print("Done")


def print_res_df(diabetes_score, sclerosis_score, depression_score, title):
    print(title)
    res_df = pd.DataFrame(pd.concat([diabetes_score, sclerosis_score, depression_score]), columns=['accuracy', 'f1_score', 'precision', 'recall'])
    avg = res_df.mean().apply(lambda x: round(x, 2))
    res_df.loc['average'] = avg
    print(res_df)
    for key in ['cm', 'report']:
        if key in diabetes_score:
            print()
            print_key('Diabetes', diabetes_score, key)
            print_key('Sclerosis', sclerosis_score, key)
            print_key('Depression', depression_score, key)
    print()

def print_key(community, score, key):
    print(community)
    print(score.iloc[0][key])
    print()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='UMLS Entity Linking Evaluator')
        parser.add_argument('data_dir', help='Path to the input directory.')
        parsed_args = parser.parse_args(sys.argv[1:])
        data_dir = parsed_args.data_dir
        print(f"Got data_dir: {data_dir}")
    else:
        from config import data_dir, FINAL_LABELS_COL, CHEMICAL_OR_DRUG, DISORDER

    training_dataset_dir = data_dir + r"contextual_relevance\extracted_training_dataset"
    output_models_dir = data_dir + r"contextual_relevance\output_models"
    labels_dir = data_dir + r'manual_labeled_v2\doccano\merged_output'
    results_dir = data_dir + r'results'

    main()
