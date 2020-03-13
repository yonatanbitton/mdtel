import os
import json

from config import data_dir

labels_dir = data_dir + r'manual_labeled_v2\doccano'

def annotated_by_user(annotations, user_num):
    for ann in annotations:
        if ann['user']==user_num:
            return True
    return False

def main():
    print_comm_data(community='sclerosis')
    print_comm_data(community='diabetes')
    print_comm_data(community='depression')
    print("Yo")


def print_comm_data(community):
    with open(labels_dir + os.sep + community + "_export.json", encoding='utf-8') as f:
        lines = [json.loads(x) for x in f.readlines()]
    lines_with_user_5 = lines_with_user_6 = 0
    lines_with_both_user = 0
    for l in lines:
        if l['annotations'] == []:
            continue
        annotations = l['annotations']
        user_5_in = annotated_by_user(annotations, 5)
        if user_5_in:
            lines_with_user_5 += 1
        user_6_in = annotated_by_user(annotations, 6)
        if user_6_in:
            lines_with_user_6 += 1
        if user_5_in and user_6_in:
            lines_with_both_user += 1
        # if user_6_in:
        #     print(annotations)
    print(f"comm {community}, lines_with_user_5: {lines_with_user_5}, lines_with_user_6: {lines_with_user_6}")
    print(f"comm {community}, lines_with_both_user: {lines_with_both_user}")
    print("\n")


if __name__ == '__main__':
    main()