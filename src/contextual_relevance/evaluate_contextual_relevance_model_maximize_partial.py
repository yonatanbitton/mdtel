import argparse
import json
import os
import sys
from statistics import mode

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, precision_score, \
    cohen_kappa_score
from sklearn.model_selection import train_test_split

from config import CHEMICAL_OR_DRUG, DISORDER
from high_recall_matcher_posts_level import words_similarity
from utils import SequenceTagger

module_path = os.path.abspath(os.path.join('..', '..', os.getcwd()))
sys.path.append(module_path)

add_extra_fns = False
from collections import namedtuple
Entity = namedtuple("Entity", "e_type start_offset end_offset")
from contextual_relevance.ner_eval import compute_metrics
from copy import deepcopy
from contextual_relevance.ner_eval import compute_precision_recall_wrapper


def evaluate_community(community):
    print(f"*** community: {community} ***")
    community_df, labels_df = get_data(community)

    if add_extra_fns:
        disorders_number_extra_fns, chemical_or_drugs_extra_fns = get_number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list(
            community, community_df)
    else:
        disorders_number_extra_fns, chemical_or_drugs_extra_fns = 0, 0

    best_res = get_best_results(community, community_df, DISORDER, disorders_number_extra_fns, labels_df)
    print(f"Comm: {community}")
    print(best_res['res_df'])
    p_res_dir = r'D:\ThesisResources\OHCsProject_Resources\Partial Results'
    best_res['res_df'].to_excel(p_res_dir + os.sep + community + "_partial_res.xlsx")


def get_data(community):
    community_df = pd.read_csv(training_dataset_dir + os.sep + community + ".csv")
    labels_df = pd.read_csv(labels_dir + os.sep + community + "_labels.csv")
    labels_df[FINAL_LABELS_COL] = labels_df[FINAL_LABELS_COL].apply(lambda x: json.loads(x))

    print(f"Before filter tok prob: {len(community_df)}")
    community_df = community_df[community_df['file_name'].isin(set(labels_df['file_name']))]
    print(f"After filter tok prob: {len(community_df)}")

    print(f"Number of instances: {len(community_df)}, labels distribution:")
    print(community_df['yi'].value_counts())
    return community_df, labels_df


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


def calc_metrics_for_row(row):
    y_true = [x[1] for x in row['bio_tags']['words_and_tags']]
    y_pred = [x[1] for x in row['prediction_bio_tags']['words_and_tags']]
    assert len(y_true) == len(y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return kappa


def get_entity_named_tuple(community, lst):
    all_ents = []
    for t in lst:
        label = t['label']
        if not type(label) == str:
            label = label_numbers_for_each_community[community][label]
        ent = Entity(label, t['start_offset'], t['end_offset'])
        all_ents.append(ent)
    return all_ents


def create_bio_tag(community, community_df, chemicals_experiment_data, disorders_experiment_data, labels_df):
    disorders_seq_data = disorders_experiment_data['initial_results']['X_test_seq_data']
    chemicals_seq_data = chemicals_experiment_data['initial_results']['X_test_seq_data']
    dis_fnames = set(disorders_seq_data['file_name'])
    chem_fnames = set(chemicals_seq_data['file_name'])
    fnames_inner_intersections = dis_fnames.intersection(chem_fnames)
    dis_fnames_no_chems = dis_fnames.difference(chem_fnames)
    chems_fnames_no_dis = chem_fnames.difference(dis_fnames)
    test_filenames = fnames_inner_intersections.union(dis_fnames_no_chems, chems_fnames_no_dis)

    relevant_cols = ['cand_match', 'umls_match', 'match_eng', 'file_name', 'curr_occurence_offset', 'all_match_occ',
                     'yi', 'semantic_type']
    community_df = community_df[relevant_cols]
    test_comm_df = community_df[community_df['file_name'].isin(test_filenames)]
    test_labels_df = get_df_with_preds_and_annotations(chemicals_seq_data, community, disorders_seq_data, labels_df,
                                                       test_comm_df, test_filenames)
    pred_named_tuples = list(test_labels_df['prediction_named_tuple'].values)
    true_named_tuples = list(test_labels_df['annotation_named_tuple'].values)
    # get_partial_results(community, pred, true)
    annotation_tagger = SequenceTagger(community)
    pred_tagger = SequenceTagger(community, predictor=True)

    all_y_true = []
    all_y_pred = []

    for idx, row in test_labels_df.iterrows():
        annotations = row[FINAL_LABELS_COL]
        predictions = row['prediction_labels']
        annotation_terms = [m['term'] for m in annotations]
        prediction_terms = [m['term'] for m in predictions]
        annotated_not_predicted = set(annotation_terms).difference(prediction_terms)
        predicted_not_annotated = set(prediction_terms).difference(annotation_terms)
        # if len(annotated_not_predicted) > 0 or len(predicted_not_annotated) > 0:
            # print("* Diff *")
            # print(f"Annotated not predicted: ")
            # print(annotated_not_predicted)
            # print(f"Predicted not annotated: ")
            # print(predicted_not_annotated)
        annotation_bio_tags = annotation_tagger.get_bio_tags(row, FINAL_LABELS_COL)
        prediction_bio_tags = pred_tagger.get_bio_tags(row, 'prediction_labels')
        post_y_true = [x[1] for x in annotation_bio_tags['words_and_tags']]
        post_y_pred = [x[1] for x in prediction_bio_tags['words_and_tags']]
        assert len(post_y_true) == len(post_y_pred)
        all_y_true.append(post_y_true)
        all_y_pred.append(post_y_pred)

    y_true = all_y_true
    y_pred = all_y_pred

    # from seqeval.metrics import accuracy_score as seq_eval_acc
    # from seqeval.metrics import classification_report
    # from seqeval.metrics import f1_score as seq_eval_f1
    # token_f1_score = seq_eval_f1(y_true, y_pred)
    # token_acc_score = seq_eval_acc(y_true, y_pred)
    # token_classification_report = classification_report(y_true, y_pred)

    from sklearn_crfsuite import metrics
    labels = ['B-D', 'I-D', 'B-C', 'I-C']
    token_acc_score = metrics.flat_accuracy_score(y_true, y_pred)
    token_recall_score = metrics.flat_recall_score(y_true, y_pred, average='weighted', labels=labels)
    token_f1_score = metrics.flat_f1_score(y_true, y_pred, average='weighted', labels=labels)

    print(f"token_f1_score: {token_f1_score}, token_acc_score: {token_acc_score}, token_recall_score: {token_recall_score}")

    print("Done")

def get_df_with_preds_and_annotations_all_types(X_test, community, labels_df):
    test_filenames = set(X_test['file_name'].values)
    grouped = X_test.groupby('file_name')
    labels_per_post = {}
    for idx, post_df in grouped:
        post_file_name = post_df['file_name'].iloc[0]
        X_filename = X_test[X_test['file_name'] == post_file_name]
        X_filename = X_filename[X_filename['hard_y_pred'] == 1]
        X_filename_labels = get_labels_for_semantic_type(X_filename)
        if post_file_name in labels_per_post:
            print(f"Problem. Double. {community}")
        labels_per_post[post_file_name] = X_filename_labels
    test_labels_df = labels_df[labels_df['file_name'].isin(test_filenames)]
    test_labels_df['prediction_labels'] = test_labels_df['file_name'].apply(
        lambda post_file_name: labels_per_post[post_file_name])
    test_labels_df['prediction_labels'] = test_labels_df.apply(lambda row: assert_terms_in_place(row), axis=1)
    test_labels_df['prediction_named_tuple'] = test_labels_df['prediction_labels'].apply(
        lambda lst: get_entity_named_tuple(community, lst))
    test_labels_df['annotation_named_tuple'] = test_labels_df['merged_inner_and_outer'].apply(
        lambda lst: get_entity_named_tuple(community, lst))
    return test_labels_df


def get_df_with_preds_and_annotations(chemicals_seq_data, community, disorders_seq_data, labels_df, test_comm_df,
                                      test_filenames):
    grouped = test_comm_df.groupby('file_name')
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
            print(f"Problem. Double. {community}")
        labels_per_post[post_file_name] = joint_labels
    test_labels_df = labels_df[labels_df['file_name'].isin(test_filenames)]
    test_labels_df['prediction_labels'] = test_labels_df['file_name'].apply(
        lambda post_file_name: labels_per_post[post_file_name])
    test_labels_df['prediction_labels'] = test_labels_df.apply(lambda row: assert_terms_in_place(row), axis=1)
    test_labels_df['prediction_named_tuple'] = test_labels_df['prediction_labels'].apply(
        lambda lst: get_entity_named_tuple(community, lst))
    test_labels_df['annotation_named_tuple'] = test_labels_df['merged_inner_and_outer'].apply(
        lambda lst: get_entity_named_tuple(community, lst))
    return test_labels_df


def get_partial_results(community, pred, true):
    tags = ['Disorder', 'Chemical or drug']
    # m = compute_metrics(true[1], pred[1], tags=['Disorder', 'Chemical or drug'])
    test_sents_labels = true
    y_pred = pred
    metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                       'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0, 'precision': 0, 'recall': 0}
    # overall results
    results = {'strict': deepcopy(metrics_results),
               'ent_type': deepcopy(metrics_results),
               'partial': deepcopy(metrics_results),
               'exact': deepcopy(metrics_results)
               }
    # results aggregated by entity type
    evaluation_agg_entities_type = {e: deepcopy(results) for e in tags}
    for true_ents, pred_ents in zip(test_sents_labels, y_pred):

        # compute results for one message
        tmp_results, tmp_agg_results = compute_metrics(
            true_ents, pred_ents, tags
        )

        # print(tmp_results)

        # aggregate overall results
        for eval_schema in results.keys():
            for metric in metrics_results.keys():
                results[eval_schema][metric] += tmp_results[eval_schema][metric]

        # Calculate global precision and recall

        results = compute_precision_recall_wrapper(results)

        # aggregate results by entity type

        for e_type in tags:

            for eval_schema in tmp_agg_results[e_type]:

                for metric in tmp_agg_results[e_type][eval_schema]:
                    evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][
                        metric]

            # Calculate precision recall at the individual entity level

            evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(
                evaluation_agg_entities_type[e_type])
    res_df = pd.DataFrame(results)
    return res_df


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


def get_best_results(community, community_df, semantic_type, extra_fns, labels_df):
    semantic_type_df = community_df[community_df['semantic_type'] == semantic_type]
    selected_feats = ['match_count', 'match_freq', 'pred_3_window', 'pred_6_window', 'pred_10_window', 'relatedness']
    # selected_feats = ['match_freq', 'pred_3_window', 'pred_6_window', 'relatedness']

    best_joint_score = 0
    best_experiment_data = None
    n_estimators = 50
    test_size = 0.25

    model = RandomForestClassifier(n_estimators=n_estimators)
    X = semantic_type_df
    all_filenames = X['file_name'].values
    all_filenames_nodups = list(set(all_filenames))
    filenames_train, filenames_test = train_test_split(all_filenames_nodups, test_size=test_size, shuffle=False)

    X_train = X[X['file_name'].isin(filenames_train)]
    y_train = X_train['yi']
    X_test = X[X['file_name'].isin(filenames_test)]
    y_test = X_test['yi']
    X_train_matrix = X_train[selected_feats]
    X_test_matrix = X_test[selected_feats]
    model.fit(X_train_matrix, y_train)
    y_pred = [x[1] for x in model.predict_proba(X_test_matrix)]

    for threshold in [i / 10 for i in list(range(1, 10))]:
        experiment_data, score = get_results_for_threshold(threshold, community, community_df, X_test, y_test,
                                                           y_pred, test_size, extra_fns, labels_df)

        if score > best_joint_score:
            best_joint_score = score
            best_experiment_data = experiment_data

    # community_score = print_best_experiment_data(community, best_experiment_data, semantic_type)
    return best_experiment_data


def get_number_of_items_labeled_as_positive_but_wasnt_at_high_recall_list(community, community_df):
    # Adding misses of high recall matcher
    labels_df = pd.read_csv(labels_dir + os.sep + community + "_labels.csv")
    labels_df[FINAL_LABELS_COL] = labels_df[FINAL_LABELS_COL].apply(json.loads)

    labels_df = add_semantic_type_cols(labels_df, community)

    chemical_or_drugs_extra_fns = get_misclassifications_numbers_for_semantic_type(community_df, labels_df,
                                                                                   CHEMICAL_OR_DRUGS_COL,
                                                                                   CHEMICAL_OR_DRUG)
    disorders_number_extra_fns = get_misclassifications_numbers_for_semantic_type(community_df, labels_df,
                                                                                  DISORDERS_COL, DISORDER)

    return disorders_number_extra_fns, chemical_or_drugs_extra_fns


def get_misclassifications_numbers_for_semantic_type(community_df, labels_df, annotated_col, semantic_type):
    semantic_type_all_high_recall_matches = community_df[community_df['semantic_type'] == semantic_type][
        'cand_match'].values

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
    labels_df_community_name = labels_df[FINAL_LABELS_COL].apply(
        lambda lst: [m for m in lst if m['term'] == comm_to_heb[community]])
    community_name_user_values = [lst[0]['label'] for lst in labels_df_community_name.values if lst != []]
    print(f"comm label names")
    disorders_number = mode(community_name_user_values)
    different_num_series = labels_df[FINAL_LABELS_COL].apply(
        lambda lst: [m for m in lst if m['label'] != disorders_number])
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
        x_in_all_positive_labeled_terms = x in all_positive_labeled_terms
        if not x_in_all_positive_labeled_terms:
            for y in all_positive_labeled_terms:
                if words_similarity(x, y) > SIMILARITY_THRESHOLD or subterm_is_inside(x, y):
                    x_in_all_positive_labeled_terms = True
                    break
        if not x_in_all_positive_labeled_terms:
            missed_items.append(x)
    return missed_items


def get_results_for_threshold(threshold, community, community_df, X_test, y_test, y_pred, test_size, extra_fns, labels_df):
    hard_y_pred = [int(x > threshold) for x in y_pred]
    y_test = list(y_test.values)

    cm_examples = get_confusion_matrix_examples(X_test, community_df, hard_y_pred, y_test)

    X_test['hard_y_pred'] = hard_y_pred
    test_labels_df = get_df_with_preds_and_annotations_all_types(X_test, community, labels_df)

    pred_named_tuples = list(test_labels_df['prediction_named_tuple'].values)
    true_named_tuples = list(test_labels_df['annotation_named_tuple'].values)

    res_df = get_partial_results(community, pred_named_tuples, true_named_tuples)
    cm = confusion_matrix(y_test, hard_y_pred)

    recall_partial = res_df.at['recall', 'partial']

    score = recall_partial
    experiment_data = {'res_df': res_df,  'community': community,
                       'confusion_matrix': cm}
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
    evaluate_community('sclerosis')
    evaluate_community('diabetes')
    evaluate_community('depression')



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
