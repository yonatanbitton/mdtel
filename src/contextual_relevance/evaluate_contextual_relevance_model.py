import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

training_dataset_dir = r'E:\mdtel_data\data\contextual_relevance\training_dataset'
output_models_dir = r'E:\mdtel_data\data\contextual_relevance\output_models'

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
    data_to_return = best_experiment_data
    m = best_experiment_data[community].pop('model')
    print(best_experiment_data)
    return data_to_return


def get_results_for_threshold(community, community_df, n_estimators, selected_feats, test_size, threshold):
    model = RandomForestClassifier(n_estimators=n_estimators)
    X = community_df[selected_feats]
    y = community_df['yi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
    model.fit(X_train, y_train)
    y_pred = [x[1] for x in model.predict_proba(X_test)]
    hard_y_pred = [x > threshold for x in y_pred]
    f1_score_score = f1_score(y_test, hard_y_pred)
    cm = confusion_matrix(y_test, hard_y_pred)
    precision_score_score = precision_score(y_test, hard_y_pred)
    recall_score_score = recall_score(y_test, hard_y_pred)
    acc = accuracy_score(y_test, hard_y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    score = f1_score_score + roc_auc
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    experiment_data = {community: {'cm': cm, 'f1_score': f1_score_score,
                                   'roc_auc': roc_auc, 'acc': acc,
                                   'precision_score': precision_score_score,
                                   'recall': recall_score_score,
                                   'threshold': threshold,
                                   'model': model}}
    return experiment_data, score

def main():
    diabetes_model_data = evaluate_community('diabetes')
    sclerosis_model_data = evaluate_community('sclerosis')
    depression_model_data = evaluate_community('depression')

    trained_models = {'diabetes': diabetes_model_data, 'sclerosis': sclerosis_model_data,
                      'depression': depression_model_data}

    with open(output_models_dir + os.sep + "trained_models.pickle", 'wb') as f:
        pickle.dump(trained_models, f)

if __name__ == '__main__':
    main()

print("DONE")
