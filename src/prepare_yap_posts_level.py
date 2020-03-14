import json
import os

import pandas as pd

from config import data_dir
from contextual_relevance.extract_dataset_with_feats.extract_relatedness_features import \
    get_dataframes_of_words_and_lemmas, get_all_words_and_lemmas_from_df
from contextual_relevance.extract_dataset_with_feats.yap.yap_api import YapApi

raw_tokenized_dir = data_dir + r"high_recall_matcher\posts\lemlda\raw_tokenized"
output_dir = data_dir + r"high_recall_matcher\posts\lemlda"
yap_dataframes_output = output_dir + os.sep + 'yap_dataframes'
dep_trees_output = yap_dataframes_output + os.sep + "dep_trees"
md_lattices_output = yap_dataframes_output + os.sep + "md_lattices"

yap = YapApi()
ip = '127.0.0.1:8000'

general_count = 0

def parse_community(community):
    print(f"Handling community: {community}")
    comm_dir = raw_tokenized_dir + os.sep + community
    all_df_rows = []
    for fname in os.listdir(comm_dir):
        fpath = comm_dir + os.sep + fname
        f_name_str = fname.split(".txt")[0]
        handle_fpath(fpath, f_name_str, all_df_rows, community)

    comm_df = pd.DataFrame(all_df_rows, columns=list(all_df_rows[0].keys()))
    comm_df['words_and_lemmas'] = comm_df['words_and_lemmas'].apply(lambda x: json.dumps(x, ensure_ascii=False))
    comm_df.to_excel(output_dir + os.sep + community + "_posts.xlsx", index=False)
    print(f"Finished with community {community}")

def get_yap_features(post_example, community, file_name):
    tokenized_text, segmented_text, lemmas, dep_tree, md_lattice, ma_lattice = yap.run(post_example, ip)
    dataframes = get_dataframes_of_words_and_lemmas(md_lattice)
    if dataframes == []:
        return {}

    dep_tree_comm_dir = dep_trees_output + os.sep + community
    md_lattice_comm_dir = md_lattices_output + os.sep + community
    if not os.path.exists(dep_tree_comm_dir):
        os.mkdir(dep_tree_comm_dir)
    if not os.path.exists(md_lattice_comm_dir):
        os.mkdir(md_lattice_comm_dir)

    dep_tree_file_path = dep_tree_comm_dir + os.sep + file_name + ".csv"
    md_lattice_file_path = md_lattice_comm_dir + os.sep + file_name + ".csv"

    dep_tree.to_csv(dep_tree_file_path, encoding='utf-8-sig', index=False)
    md_lattice.to_csv(md_lattice_file_path, encoding='utf-8-sig', index=False)

    words_and_lemmas = get_all_words_and_lemmas_from_df(dataframes)

    yap_d = {'tokenized_text': tokenized_text, 'segmented_text': segmented_text, 'lemmas': lemmas, 'words_and_lemmas': words_and_lemmas}

    return yap_d


def handle_fpath(fpath, file_name, all_df_rows, community):
    global general_count
    with open(fpath, encoding='utf-8') as f:
        post_txt_lines = [x.rstrip('\n') for x in f.readlines()]
    post_txt_lines = [l for l in post_txt_lines if l != '']
    post_txt_dots = "\n".join(post_txt_lines)
    if '\xa0' in post_txt_dots:
        post_txt_dots = post_txt_dots.replace('\xa0', ' ')
    if post_txt_dots == "" or post_txt_dots == " ":
        return
    general_count += 1
    post_yap_features = get_yap_features(post_txt_dots, community, file_name)

    # post_d = {'file_name': file_name, 'post_words_and_lemmas': json.dumps(post_words_and_lemmas, ensure_ascii=False),
    #           'post_txt': post_txt_dots, 'post_dep_tree': post_dep_tree}
    post_d = {'file_name': file_name, 'post_txt': post_txt_dots, **post_yap_features}

    if general_count % 50 == 0 or "שלום שושי" in post_txt_dots:
        if "שלום שושי" in post_txt_dots:
            print(f"שלום שושי!!!")
        print(post_d)
    all_df_rows.append(post_d)


def main():
    # fpath = r"E:\mdtel_data\data\high_recall_matcher\posts\lemlda\raw_tokenized\sclerosis\15.txt"
    # handle_fpath_newlines(fpath)
    # parse_community('sclerosis')
    # parse_community('diabetes')
    parse_community('depression')

if __name__ == '__main__':
    main()
