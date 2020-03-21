import difflib
import json
import os
import sys

import pandas as pd

module_path = os.path.abspath(os.path.join('..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, SIMILARITY_THRESHOLD

input_dir = data_dir + r"contextual_relevance\initialized_training_dataset"
yap_dataframes_dir = data_dir + r"high_recall_matcher\posts\lemlda\yap_dataframes"
dep_trees_dir = yap_dataframes_dir + os.sep + "dep_trees"
md_lattices_dir = yap_dataframes_dir + os.sep + "md_lattices"

output_dir = data_dir + r"contextual_relevance\yap_features"

number_of_sorts = 0

def handle_community(community):
    print(f"Yap features extractor, community: {community}")
    comm_df = pd.read_csv(input_dir + os.sep + community + ".csv")
    comm_df['txt_words'] = comm_df['txt_words'].apply(json.loads)
    comm_df['occurences_indexes_in_txt_words'] = comm_df['occurences_indexes_in_txt_words'].apply(json.loads)

    comm_dep_trees_dir = dep_trees_dir + os.sep + community
    comm_md_lattices_dir = md_lattices_dir + os.sep + community

    all_rows_with_yap = []

    global number_of_sorts
    number_of_sorts = 0

    for row_idx, row in comm_df.iterrows():
        full_fname = str(row['file_name']) + ".csv"
        dep_tree_d = get_dep_tree_features(comm_dep_trees_dir, full_fname, row)
        md_lattice_d = get_md_lattice_features(comm_md_lattices_dir, full_fname, row)

        yap_d = {**row.to_dict(), **dep_tree_d, **md_lattice_d}
        all_rows_with_yap.append(yap_d)

        # if row_idx % 10 == 0:
        #     print(f"row_idx: {row_idx}, out of: {len(comm_df)}, match: {row['umls_match']}, number_of_sorts: {number_of_sorts}")

    comm_df_yap = pd.DataFrame(all_rows_with_yap)
    comm_df_yap.to_csv(output_dir + os.sep + community + "_output.csv", index=False, encoding='utf-8-sig')


def get_md_lattice_features(comm_md_lattices_dir, full_fname, row):
    cand_match = row['cand_match']
    umls_match = row['umls_match']
    md_lattice = pd.read_csv(comm_md_lattices_dir + os.sep + full_fname)
    correct_md_lattice_row = get_correct_row_from_df(md_lattice, cand_match, umls_match, row)
    gen = correct_md_lattice_row['gen']
    pos = correct_md_lattice_row['pos']
    tense = correct_md_lattice_row['tense']
    md_lattice_d = {'gen': gen, 'pos': pos, 'tense': tense}
    return md_lattice_d

def get_dep_tree_features(comm_dep_trees_dir, full_fname, row):
    cand_match = row['cand_match']
    umls_match = row['umls_match']
    dep_tree = pd.read_csv(comm_dep_trees_dir + os.sep + full_fname)
    correct_dep_tree_row = get_correct_row_from_df(dep_tree, cand_match, umls_match, row)
    dep_part = correct_dep_tree_row['dependency_part']
    dep_tree_d = {'dep_part': dep_part}
    return dep_tree_d

def get_correct_row_from_df(df, cand_match, umls_match, row):
    # global comm_duplicates
    number_of_match = row['occurences_indexes_in_txt_words'].index(row['match_occurence_idx_in_txt_words'])
    cand_match_parts = cand_match.split(" ")
    match_len = len(cand_match_parts)
    cand_match_first_term = cand_match_parts[0]
    all_possible_word_rows = df[df['word'].apply(lambda x: words_similarity(x, cand_match_first_term) > SIMILARITY_THRESHOLD)]

    if len(row['occurences_indexes_in_txt_words']) == len(all_possible_word_rows):
        correct_row = all_possible_word_rows.iloc[number_of_match]
    else:
        all_possible_lemma_rows = df[df['lemma'].apply(lambda x: words_similarity(x, cand_match_first_term) > SIMILARITY_THRESHOLD)]
        if len(row['occurences_indexes_in_txt_words']) == len(all_possible_lemma_rows):
            correct_row = all_possible_lemma_rows.iloc[number_of_match]
        else:
            try:
                best_idx_with_prefix = try_to_find_with_prefix(all_possible_word_rows, cand_match, df, match_len)
            except Exception as ex:
                best_idx_with_prefix = None
            if best_idx_with_prefix:
                correct_row = df.loc[best_idx_with_prefix]
            else:
                best_idx_with_next_word = try_to_find_with_next_term(all_possible_word_rows, cand_match, df, match_len)
                if best_idx_with_next_word:
                    correct_row = df.loc[best_idx_with_next_word]
                else:
                    global number_of_sorts
                    number_of_sorts += 1
                    df['word_sim_to_cand_match'] = df['word'].apply(lambda x: words_similarity(x, cand_match_first_term))
                    df.sort_values(by='word_sim_to_cand_match', ascending=False, inplace=True)
                    correct_row = df.iloc[0]

    return correct_row


def try_to_find_with_next_term(all_possible_word_rows, cand_match, df, match_len):
    best_idx_with_next_word = None
    best_sim = 0
    for idx in all_possible_word_rows.index:
        term_with_next_word = " ".join(df['word'].loc[idx: idx + match_len - 1])
        sim = words_similarity(term_with_next_word, cand_match)
        if sim > SIMILARITY_THRESHOLD and sim > best_sim:
            best_sim = sim
            best_idx_with_next_word = idx
    return best_idx_with_next_word

def try_to_find_with_prefix(all_possible_word_rows, cand_match, df, match_len):
    best_idx_with_prefix = None
    for idx in all_possible_word_rows.index:
        if match_len == 1:
            prefix_cand = "".join(df['word'].loc[idx - 1:idx].values)
        else:
            prefix_cand = df['word'].loc[idx - 1] + "" + df['word'].loc[idx] + " " + df['word'].loc[idx + 1]
        if prefix_cand == cand_match:
            best_idx_with_prefix = idx
            break
    return best_idx_with_prefix


def words_similarity(a, b):
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio()

if __name__ == '__main__':
    handle_community("sclerosis")
    handle_community("diabetes")
    handle_community("depression")

    print("Yap features extractor - Done.")