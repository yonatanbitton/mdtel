import json
import os

import pandas as pd

from config import data_dir

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
    len_before_drop_dups = len(comm_df)
    comm_df = comm_df.drop_duplicates(subset=['post_txt', 'lemmas'])
    print(f"Length before drop dups: {len_before_drop_dups}, after: {len(comm_df)}")
    comm_df.to_excel(output_dir + os.sep + community + "_posts.xlsx", index=False)
    print(f"Finished with community {community}")

def get_yap_features(post_example, community, file_name):
    post_example = post_example.replace("\n", " ").replace(".", " ").replace(",", " ")
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
    post_d = {'file_name': file_name, 'post_txt': post_txt_dots, **post_yap_features}

    all_df_rows.append(post_d)

def get_words_and_lemmas(post_example):
    tokenized_text, segmented_text, lemmas, dep_tree, md_lattice, ma_lattice = yap.run(post_example, ip)
    dataframes = get_dataframes_of_words_and_lemmas(md_lattice)
    if dataframes == []:
        return []

    words_and_lemmas = get_all_words_and_lemmas_from_df(dataframes)
    return words_and_lemmas

def get_dataframes_of_words_and_lemmas(md_lattice):
    if 'num_last' not in md_lattice:
        return []
    md_lattice['num_last'] = md_lattice['num_last'].apply(lambda x: int(x))
    indexes_of_starts = find_indexes_of_start(md_lattice)
    if len(indexes_of_starts) == 1:
        dataframes = [md_lattice]
    else:
        dataframes = build_dataframes(indexes_of_starts, md_lattice)
    return dataframes

def build_dataframes(indexes_of_starts, md_lattice):
    pairs = list(zip(indexes_of_starts, indexes_of_starts[1:]))
    dataframes = []
    for pair in pairs:
        df = md_lattice.iloc[pair[0]:pair[1]]
        dataframes.append(df)
    last_df = md_lattice.iloc[pairs[-1][1]:]
    dataframes.append(last_df)
    return dataframes

def find_indexes_of_start(md_lattice):
    indexes_of_starts = []
    for idx, val in enumerate(md_lattice['num_last'].values):
        if val == 1:
            indexes_of_starts.append(idx)
    return indexes_of_starts


def max_len(s):
    return max(s, key=len)

def get_all_words_and_lemmas_from_df(dataframes):
    all_words_and_lemmas = []
    for df in dataframes:
        words_and_lemmas_groups = df.groupby('num_last').agg({'lemma': max_len, 'word': max_len})
        words_and_lemmas_groups = words_and_lemmas_groups.sort_values(by=['num_last'])
        words_and_lemmas = [x['lemma'] + " " + x['word'] for x in
                            words_and_lemmas_groups.to_dict('index').values()]
        all_words_and_lemmas += words_and_lemmas
    return all_words_and_lemmas


def main():
    parse_community('depression')
    parse_community('sclerosis')
    parse_community('diabetes')

if __name__ == '__main__':
    main()
