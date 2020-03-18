import os
import json
import pandas as pd
from config import data_dir

labels_dir = data_dir + r'manual_labeled_v2\doccano'
labels_csv_files = labels_dir + os.sep + "csv_files"

def annotated_by_user(annotations, user_num):
    for ann in annotations:
        if ann['user']==user_num:
            return True
    return False

def main():
    # print_comm_data(community='sclerosis')
    # print_comm_data(community='diabetes')
    print_comm_data(community='depression_merged')
    print("Done")


def print_comm_data(community):
    with open(labels_dir + os.sep + community + "_export.json", encoding='utf-8') as f:
        lines = [json.loads(x) for x in f.readlines()]
    lines_with_user_5 = lines_with_user_6 = 0
    lines_with_both_user = 0

    all_line_texts = []
    all_user_5_labels = []
    all_user_6_labels = []
    in_user_6_but_not_in_5 = 0
    in_user_5_but_not_in_6 = 0

    for l in lines:
        # if l['annotations'] == []:
        #     continue
        annotations = l['annotations']
        user_5_in = annotated_by_user(annotations, 5)
        if user_5_in:
            lines_with_user_5 += 1
        user_6_in = annotated_by_user(annotations, 6)
        if user_6_in:
            lines_with_user_6 += 1
        if user_5_in and user_6_in:
            lines_with_both_user += 1

        all_line_texts.append(l['text'])

        line_annotations = l['annotations']
        if line_annotations == []:
            all_user_5_labels.append(line_annotations)
            all_user_6_labels.append(line_annotations)
        else:
            user_5_labels = []
            user_6_labels = []
            for ann in line_annotations:
                annotated_term = l['text'][ann['start_offset']:ann['end_offset']]
                if ann['user'] == 5:
                    user_5_labels.append(annotated_term)
                elif ann['user'] == 6:
                    user_6_labels.append(annotated_term)
                else:
                    print(f"Unknown annotation! {ann}")
            if user_5_labels != [] and user_6_labels == []:
                in_user_5_but_not_in_6 += 1
            if user_5_labels == [] and user_6_labels != []:
                in_user_6_but_not_in_5 += 1
            all_user_5_labels.append(user_5_labels)
            all_user_6_labels.append(user_6_labels)
        # if user_6_in:
        #     print(annotations)

    print("\n\n")
    print(f'lines_with_user_5: {lines_with_user_5}, lines_with_user_6: {lines_with_user_6}, lines_with_both_user: {lines_with_both_user}')
    print("\n\n")
    print(f"in_user_5_but_not_in_6: {in_user_5_but_not_in_6}, in_user_6_but_not_in_5: {in_user_6_but_not_in_5}")

    labels_df = pd.DataFrame()
    labels_df['text'] = all_line_texts
    labels_df['user_5_labels'] = all_user_5_labels
    labels_df['user_6_labels'] = all_user_6_labels

    all_user_5_labels = list(labels_df['user_5_labels'].values)
    last_relevant_idx_user_5 = -1
    last_relevant_idx_user_6 = -1
    for idx in range(len(all_user_5_labels)):
        rest_lst_user_5 = all_user_5_labels[idx:]
        rest_lst_user_6 = all_user_6_labels[idx:]
        if all(x==[] for x in rest_lst_user_5) and last_relevant_idx_user_5 == -1:
            # print(f"comm: {community}, last tagged 5 : {idx}")
            last_relevant_idx_user_5 = idx
            # break
        if all(x==[] for x in rest_lst_user_6) and last_relevant_idx_user_6 == -1:
            # print(f"comm: {community}, last tagged 6: {idx}")
            last_relevant_idx_user_6 = idx
            # break
    print(f"comm: {community}, last tagged 5: {last_relevant_idx_user_5}, last tagged 6: {last_relevant_idx_user_6}")
    return
    # if last_relevant_idx != -1:
    #     labels_df = labels_df.head(last_relevant_idx)

    labels_df.to_csv(labels_csv_files + os.sep + community + "_labels.csv", index=False, encoding='utf-8-sig')

    unannotated = labels_df[labels_df['user_5_labels'].apply(lambda x: x == [])]
    unannotated.to_csv(labels_csv_files + os.sep + community + "_unannotated.csv", index=False, encoding='utf-8-sig')
    # print(f'unannotated: {unannotated.shape}')

    annotated = labels_df[labels_df['user_5_labels'].apply(lambda x: x != [])]
    annotated.to_csv(labels_csv_files + os.sep + community + "_annotated.csv", index=False, encoding='utf-8-sig')
    # print(f'unannotated: {unannotated.shape}')

    print(f"comm {community}, lines_with_user_5: {lines_with_user_5}, lines_with_user_6: {lines_with_user_6}")
    print(f"comm {community}, lines_with_both_user: {lines_with_both_user}")
    print("\n")


if __name__ == '__main__':
    main()