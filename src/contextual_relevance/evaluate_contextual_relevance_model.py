import argparse
import os
import pickle
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def evaluate_community(community):
    community_df = pd.read_csv(training_dataset_dir + os.sep + community + "_debug.csv")
    selected_feats = ['match_freq', 'pred_3_window', 'pred_6_window', 'relatedness']

    best_joint_score = 0
    best_experiment_data = None
    n_estimators = 50
    test_size = 0.20
    for threshold in [i/10 for i in list(range(1, 10))]:
        experiment_data, score = get_results_for_threshold(community, community_df, n_estimators, selected_feats,
                                                           test_size, threshold)

        if score > best_joint_score:
            best_joint_score = score
            best_experiment_data = experiment_data
    return best_experiment_data


def get_results_for_threshold(community, community_df, n_estimators, selected_feats, test_size, threshold):
    model = RandomForestClassifier(n_estimators=n_estimators)
    X = community_df[selected_feats]
    y = community_df['yi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    model.fit(X_train, y_train)
    y_pred = [x[1] for x in model.predict_proba(X_test)]
    hard_y_pred = [x > threshold for x in y_pred]
    cm = confusion_matrix(y_test, hard_y_pred)
    f1_score_score = round(f1_score(y_test, hard_y_pred), 2)
    recall_score_score = round(recall_score(y_test, hard_y_pred), 2)
    acc = round(accuracy_score(y_test, hard_y_pred), 2)
    roc_auc = round(roc_auc_score(y_test, y_pred), 2)
    score = f1_score_score
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    experiment_data = {'f1_score': f1_score_score,'roc_auc': roc_auc, 'acc': acc,
                       'recall': recall_score_score, 'community': community, 'model': model,
                       'confusion_matrix': cm}
    return experiment_data, score

def main():
    diabetes_model_data = evaluate_community('diabetes')
    sclerosis_model_data = evaluate_community('sclerosis')
    depression_model_data = evaluate_community('depression')

    res_df = pd.DataFrame([diabetes_model_data, sclerosis_model_data, depression_model_data], index=['diabetes', 'sclerosis', 'depression'], columns=['f1_score', 'roc_auc', 'recall', 'acc'])
    print(res_df)

    trained_models = {'diabetes': diabetes_model_data, 'sclerosis': sclerosis_model_data,
                      'depression': depression_model_data}

    with open(output_models_dir + os.sep + "trained_models.pickle", 'wb') as f:
        pickle.dump(trained_models, f)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Description of your app.')
        parser.add_argument('data_dir', help='Path to the input directory.')
        parsed_args = parser.parse_args(sys.argv[1:])
        data_dir = parsed_args.data_dir
        print(f"Got data_dir: {data_dir}")
    else:
        from config import data_dir

    training_dataset_dir = data_dir + r"contextual_relevance\training_dataset"
    output_models_dir = data_dir + r"contextual_relevance\output_models"

    main()

