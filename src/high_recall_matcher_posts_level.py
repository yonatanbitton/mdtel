import json
import os
import re
import sys
from collections import Counter
from copy import deepcopy
from operator import itemgetter

import pandas as pd
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher

from utils import words_similarity, word_is_english, replace_puncs

module_path = os.path.abspath(os.path.join('..', os.getcwd()))
sys.path.append(module_path)

from config import *

input_dir = data_dir + r"high_recall_matcher\posts\lemlda"
# umls_df_data_path = data_dir + r"high_recall_matcher\HEB_TO_ENG_DISORDERS_CHEMICALS_utf-8-sig.csv"
umls_df_data_path = data_dir + r"high_recall_matcher\mrconso_disorders_chemicals_merged.csv"
output_dir = data_dir + r"high_recall_matcher\output"

if DEBUG:
    print(f"*** DEBUG MODE: Taking 20 posts only ***")
else:
    print(f"Debug == False. Might take some time.")

def main():
    heb_db, eng_db, umls_data = get_umls_data()

    heb_searcher = Searcher(heb_db, CosineMeasure())
    eng_searcher = Searcher(eng_db, CosineMeasure())

    handle_community(SCLEROSIS, heb_searcher, eng_searcher, umls_data)
    # handle_community(DIABETES, heb_searcher, eng_searcher, umls_data)
    # handle_community(DEPRESSION, heb_searcher, eng_searcher, umls_data)

    print("Done")


def handle_community(chosen_community, heb_searcher, eng_searcher, umls_data):
    print(f"Handling community {chosen_community}")
    community_df = get_data(chosen_community)
    all_matches_found = []

    number_of_posts = len(community_df)

    for idx, (row_idx, row) in enumerate(community_df.iterrows()):
        if row['file_name'] != 15:
            continue
        msg_matches_found = get_english_and_hebrew_matches(eng_searcher, heb_searcher, row, umls_data)

        all_matches_found.append(json.dumps(msg_matches_found, ensure_ascii=False))
        print_stats(idx, row, msg_matches_found, number_of_posts)

    community_df['matches_found'] = all_matches_found

    if DEBUG:
        output_path = output_dir + os.sep + chosen_community + "_debug.csv"
    else:
        output_path = output_dir + os.sep + chosen_community + ".csv"

    community_df.to_csv(output_path, encoding='utf-8-sig', index=False)

    print(f"Finished with community {chosen_community}")
    print(f"Wrote to path {output_path}")


def get_english_and_hebrew_matches(eng_searcher, heb_searcher, row, umls_data):
    if str(row['tokenized_text']) == 'nan':
        return []
    english_matches_found = get_english_matches(eng_searcher, row, umls_data)
    all_hebrew_matches_found = get_hebrew_matches(heb_searcher, row, umls_data)

    # msg_matches_found = list_union(all_hebrew_matches_found, english_matches_found, post_key='union')
    msg_matches_found = list_union_by_match_occs(all_hebrew_matches_found, english_matches_found)

    return msg_matches_found


def get_hebrew_matches(heb_searcher, row, umls_data):
    all_hebrew_matches_found = []

    # for hebrew_key in ['post_txt', 'tokenized_text', 'segmented_text', 'lemmas']:
    for hebrew_key in ['segmented_text']:
        msg_key_txt = row[hebrew_key]
        matches_found_for_key = find_umls_match_fast(msg_key_txt, heb_searcher, row, hebrew_key)

        if matches_found_for_key not in [(), []]:
            # matches_found_for_key = take_highest_sim_matches(matches_found_for_key, umls_data)
            matches_found_for_key = add_heb_umls_data(matches_found_for_key, umls_data, hebrew_key)

        all_hebrew_matches_found = list_union_by_match_occs(all_hebrew_matches_found, matches_found_for_key)

    # all_hebrew_matches_found = get_matches_with_full_occs(all_hebrew_matches_found)
    return all_hebrew_matches_found

def get_english_matches(eng_searcher, row, umls_data):
    msg_eng_text = get_eng_text(row['tokenized_text'])
    english_matches_found = find_umls_match_fast(msg_eng_text, eng_searcher, row, msg_key_lang='english')
    if english_matches_found not in [set(), []]:
        english_matches_found = add_eng_umls_data(english_matches_found, umls_data, row)
    return english_matches_found

def is_good_match(sim, word_match, i, similarity_threshold):
    return len(word_match) > 3 and sim > similarity_threshold and i == len(word_match.split(" "))


def find_umls_match_fast(msg_txt, searcher, row, msg_key_lang):
    if msg_txt == "":
        return []

    ngrams = prepare_msg_ngrams(msg_txt)

    all_matches_found = []

    for gram in ngrams:
        for i in range(1, NUMBER_OF_GRAMS + 1):
            low_similarity_threshold = LOW_SINGLE_WORD_SIMILARITY_THRESHOLD if i == 1 else LOW_MULTI_WORD_SIMILARITY_THRESHOLD
            up_similarity_threshold = UP_SINGLE_WORD_SIMILARITY_THRESHOLD if i == 1 else UP_MULTI_WORD_SIMILARITY_THRESHOLD
            cand_term = " ".join(gram[:i])
            if term_is_exception(i, cand_term):
                continue
            search_result = searcher.ranked_search(cand_term, low_similarity_threshold)
            if search_result != []:
                cosine_sim, umls_match = search_result[0]  # Cosine-Sim. I can demand another sim
                sim = words_similarity(umls_match, cand_term)
                if is_good_match(sim, umls_match, i, up_similarity_threshold):
                    add_match_data(all_matches_found, cand_term, msg_key_lang, row, sim, umls_match)

    all_matches_found_with_full_occs = get_matches_with_full_occs(all_matches_found)
    return all_matches_found_with_full_occs


def get_matches_with_full_occs(all_matches_found):
    all_matches_found_with_full_occs = deepcopy(all_matches_found)
    for m in all_matches_found:
        all_occs = deepcopy(m['all_match_occ'])
        curr_occ = m['curr_occurence_offset']
        all_occs.remove(curr_occ)
        for other_occ in all_occs:
            m_copy = deepcopy(m)
            m_copy['curr_occurence_offset'] = other_occ
            if m_copy not in all_matches_found:
                all_matches_found_with_full_occs.append(m_copy)
    return all_matches_found_with_full_occs


def add_match_data(all_matches_found, cand_term, msg_key_lang, row, sim, umls_match):
    all_match_occ, cand_match_occurence_in_txt, curr_occurence_offset = get_cand_occ_in_text(all_matches_found,
                                                                                             cand_term, msg_key_lang,
                                                                                             row)
    if all_match_occ is None:
        print(f"*** all_match_occ is None! {all_match_occ}: {cand_term}")
        print(row['post_txt'])
        print(f"Curr matches")
        print(all_matches_found)
    else:
        result = {'cand_match': cand_term, 'umls_match': umls_match,
                  'cand_match_occurence_in_txt': cand_match_occurence_in_txt,
                  'sim': round(sim, 3), 'curr_occurence_offset': curr_occurence_offset,
                  'all_match_occ': all_match_occ, 'msg_key_lang': msg_key_lang}
        all_matches_found.append(result)


def get_cand_occ_in_text(all_matches_found, cand_term, msg_key_lang, row):
    if msg_key_lang in ['lemmas', 'segmented_text']:
        all_match_occ, cand_match_occurence_in_txt, curr_occurence_offset = get_occ_by_similarity(cand_term, row, msg_key_lang)
    else:
        cand_match_occurence_in_txt = cand_term
        curr_occurence_offset, all_match_occ = find_occurence_offset(all_matches_found, cand_match_occurence_in_txt,
                                                                     row, msg_key_lang)

        if (curr_occurence_offset, all_match_occ) == (None, None):
            all_match_occ, cand_match_occurence_in_txt, curr_occurence_offset = get_occ_by_similarity(cand_term, row)

    return all_match_occ, cand_match_occurence_in_txt, curr_occurence_offset


def term_is_exception(i, cand_term):
    return i == 1 and cand_term in general_exceptions and len(cand_term) in [4, 7]


def get_occ_by_similarity(cand_term, row, msg_key_lang):
    if msg_key_lang == 'lemma':
        match_occs_in_txt = find_possible_cand_matches_occurence_in_txt_by_lemma(cand_term, row)
    else:
        match_occs_in_txt = find_possible_cand_matches_occurence_in_txt_by_key(cand_term, row, msg_key_lang)

    all_match_occ, cand_match_occurence_in_txt, curr_occurence_offset = None, None, None

    if len(match_occs_in_txt) > 0:
        all_found_match_occs = {}
        for occ in match_occs_in_txt:
            match_occs = find_occs_offsets_in_txt(occ, cand_term, row['post_txt'])
            all_found_match_occs[occ] = match_occs

        all_found_occs_lst = []
        for v in all_found_match_occs.values():
            all_found_occs_lst += v

        if len(all_found_occs_lst) > 0:

            least_common_occ = get_least_common_item_from_lst(all_found_occs_lst)
            curr_occurence_offset = least_common_occ
            dict_with_match = {k: v for k, v in all_found_match_occs.items() if least_common_occ in v}
            cand_match_occurence_in_txt = list(dict_with_match.keys())[0]
            all_match_occ = list(dict_with_match.values())[0]

    return all_match_occ, cand_match_occurence_in_txt, curr_occurence_offset

def get_least_common_item_from_lst(lst):
    vals = sorted(Counter(lst).items(), key=itemgetter(1), reverse=False)
    least_common_item_and_count = vals[0]
    return least_common_item_and_count[0]

def find_occs_offsets_in_txt(cand_match_occurence_in_txt, cand_term, post_txt):
    txt_words = post_txt.split(" ")
    cand_match_occurence_in_txt_len = len(cand_match_occurence_in_txt.split(" "))
    all_match_occ = []
    for i in range(len(txt_words)):
        relevant_words = txt_words[i:i + cand_match_occurence_in_txt_len]
        if len(relevant_words) > 1 and (relevant_words[1] == '' or relevant_words[1] == ' ') and len(txt_words) > i + cand_match_occurence_in_txt_len:
                relevant_words.append(txt_words[i + cand_match_occurence_in_txt_len])
        relevant_term = " ".join(relevant_words)
        if words_similarity(relevant_term, cand_term) > 0.82 or words_similarity(relevant_term, cand_match_occurence_in_txt) > 0.82:
            occurence_offset = post_txt.index(relevant_term)
            all_match_occ.append(occurence_offset)

    return all_match_occ

def find_possible_cand_matches_occurence_in_txt_by_lemma(cand_term, row):
    match_occs_in_txt = []
    cand_term_len = len(cand_term.split(" "))
    if cand_term_len == 1:
        for l, w in [x.split(" ") for x in row['words_and_lemmas']]:
            if words_similarity(cand_term, l) > SIMILARITY_THRESHOLD:
                match_occs_in_txt.append(w)
    else:
        for i in range(len(row['words_and_lemmas']) - cand_term_len + 1):
            relevant_words_and_lemmas = row['words_and_lemmas'][i:i + cand_term_len]
            words_cand = " ".join([w.split(" ")[1] for w in relevant_words_and_lemmas])
            lemmas_cand = " ".join([w.split(" ")[0] for w in relevant_words_and_lemmas])
            if words_similarity(lemmas_cand, cand_term) > SIMILARITY_THRESHOLD:
                match_occs_in_txt.append(words_cand)
    return match_occs_in_txt


def find_possible_cand_matches_occurence_in_txt_by_segmented_text(cand_term, row):
    match_occs_in_txt = []
    segmented_text_lst = row['segmented_text'].split(" ")
    cand_term_len = len(cand_term.split(" "))
    for i in range(len(segmented_text_lst) - cand_term_len + 1):
        relevant_words = segmented_text_lst[i:i + cand_term_len]
        words_cand = " ".join(relevant_words)
        if words_similarity(words_cand, cand_term) > SIMILARITY_THRESHOLD:
            match_occs_in_txt.append(words_cand)
    return match_occs_in_txt

def find_possible_cand_matches_occurence_in_txt_by_key(cand_term, row, segmented_text):
    match_occs_in_txt = []
    tokenized_words_lst = row[segmented_text].split(" ")
    cand_term_len = len(cand_term.split(" "))
    for i in range(len(tokenized_words_lst) - cand_term_len + 1):
        relevant_words = tokenized_words_lst[i:i + cand_term_len]
        words_cand = " ".join(relevant_words)
        if words_similarity(words_cand, cand_term) > SIMILARITY_THRESHOLD:
            match_occs_in_txt.append(words_cand)
    return match_occs_in_txt


def find_occurence_offset(all_matches_found, cand_match_occurence_in_txt, row, msg_key_lang):
    post_txt = row['post_txt']
    post_txt = post_txt.lower()
    # all_match_occ = [m.start() for m in re.finditer(r"." + cand_match_occurence_in_txt, post_txt)]
    all_match_occ = [m.start() for m in re.finditer(cand_match_occurence_in_txt, post_txt)]

    curr_occurence = 0
    for m in all_matches_found:
        if m['cand_match_occurence_in_txt'] == cand_match_occurence_in_txt:
            curr_occurence += 1

    if all_match_occ == [] or len(all_match_occ) <= curr_occurence:
        return None, None

    curr_occurence_offset = all_match_occ[curr_occurence]
    return curr_occurence_offset, all_match_occ


def prepare_msg_ngrams(msg_txt):
    msg_txt = msg_txt.replace(".", " ")
    msg_words = msg_txt.split(" ")
    msg_words = [w for w in msg_words if w != "" and w != " " and (len(w) >= 2 or w.isnumeric())]
    if len(msg_words) < 3:
        msg_words.append("PAD")
    ngrams = list(zip(*[msg_words[i:] for i in range(NUMBER_OF_GRAMS)]))
    if len(ngrams) > 0:
        last_gram = ngrams[-1]
        extra_gram = last_gram[1], last_gram[2], 'PAD'
        ngrams.append(extra_gram)
        extra_gram_2 = last_gram[2], 'PAD', 'PAD'
        ngrams.append(extra_gram_2)
    return ngrams


def get_umls_data():
    umls_data = pd.read_csv(umls_df_data_path)
    print(f"Got UMLS data at length {len(umls_data)}")
    # umls_data = umls_data[~umls_data['STR'].isna()]
    # umls_data = umls_data[umls_data['LAT'].apply(lambda x: x=='ENG' or str(x) == 'nan' or x == ' ')]
    # umls_data.sort_values(by=['HEB'], inplace=True)
    # umls_data = umls_data[umls_data['STR'].apply(lambda x: len(x) > 4)]
    # umls_data = umls_data[umls_data['HEB'].apply(lambda x: len(x) > 3)]
    # umls_data.reset_index(inplace=True)

    heb_umls_list = list(umls_data['HEB'].values)
    eng_umls_list = list(umls_data[STRING_COLUMN].values)

    heb_db = DictDatabase(CharacterNgramFeatureExtractor(2))
    eng_db = DictDatabase(CharacterNgramFeatureExtractor(2))

    for heb_w in heb_umls_list:
        heb_db.add(heb_w)

    for eng_w in eng_umls_list:
        lower_eng_w = eng_w.lower()
        eng_db.add(lower_eng_w)

    return heb_db, eng_db, umls_data


def get_data(chosen_community):
    community_df = pd.read_excel(input_dir + os.sep + chosen_community + "_posts.xlsx")
    community_df['words_and_lemmas'] = community_df['words_and_lemmas'].apply(lambda x: json.loads(x) if str(x) != 'nan' else 'nan')

    # Taking only posts that have labels (Not predicting posts without labels to measure performance)
    community_df = take_only_posts_which_has_labels(chosen_community, community_df)
    community_df['post_txt'] = community_df['post_txt'].apply(lambda x: replace_puncs(x))

    return community_df


def take_only_posts_which_has_labels(chosen_community, community_df):
    labels_dir = data_dir + r'manual_labeled_v2\doccano\merged_output'
    labels_df = pd.read_csv(labels_dir + os.sep + chosen_community + "_labels.csv")
    all_labeled_texts = [x.strip() for x in list(labels_df['text'].values)]
    community_df = community_df[community_df['post_txt'].apply(lambda x: x.strip() in all_labeled_texts)]
    print(f"Number of posts: {len(community_df)}, needed: {number_of_posts_needed[chosen_community]}")
    assert abs(len(community_df) - number_of_posts_needed[chosen_community]) < 3  # Can accept small difference
    return community_df


def list_union_by_match_occs(all_hebrew_matches_found, matches_found_for_key):
    for d in matches_found_for_key:
        all_matches_with_same_umls = [m for m in all_hebrew_matches_found if m['umls_match'] == d['umls_match']]
        if not curr_match_span_intersects_with_existing_matches(d, all_matches_with_same_umls):
            all_hebrew_matches_found.append(d)
    return all_hebrew_matches_found


def curr_match_span_intersects_with_existing_matches(d, all_matches_with_same_umls):
    curr_occurence_offset = d['curr_occurence_offset']
    end_offset = curr_occurence_offset + len(d['cand_match'])
    curr_match_intersects = False
    for m in all_matches_with_same_umls:
        m_start_offset = m['curr_occurence_offset']
        m_end_offset = m_start_offset + len(m['cand_match'])
        if abs(curr_occurence_offset - m_start_offset) <= 2 and abs(end_offset - m_end_offset) <= 2:
            curr_match_intersects = True
            break
    return curr_match_intersects


def print_stats(idx, row, msg_matches_found, number_of_posts):
    if idx % 100 == 0 and idx > 0:
        print(f"idx: {idx}, out of: {number_of_posts}, total {round((idx / number_of_posts) * 100, 3)}%")
    if DEBUG or idx % 100 == 0:
        if msg_matches_found != None and msg_matches_found != []:
            print(row['tokenized_text'])
            print(msg_matches_found)
            print()


def get_semantic_type(match_tui):
    if match_tui in disorder_tuis:
        return DISORDER
    elif match_tui in chemical_or_drug_tuis:
        return CHEMICAL_OR_DRUG


def add_heb_umls_data(heb_matches, umls_data, hebrew_key):
    heb_matches_with_codes = []
    for match_d in heb_matches:
        idx_of_match_in_df = umls_data['HEB'].searchsorted(match_d['umls_match'])
        match_cui = umls_data.iloc[idx_of_match_in_df]['CUI']
        match_tui = umls_data.iloc[idx_of_match_in_df]['TUI']
        semantic_type = get_semantic_type(match_tui)
        match_eng = umls_data.iloc[idx_of_match_in_df][STRING_COLUMN]
        heb_d = {'cand_match': match_d['cand_match'], 'umls_match': match_d['umls_match'], 'sim': match_d['sim'],
                 'cui': match_cui, 'match_eng': match_eng, 'hebrew_key': hebrew_key, 'match_tui': match_tui,
                 'semantic_type': semantic_type, 'all_match_occ': match_d['all_match_occ'],
                 'curr_occurence_offset': match_d['curr_occurence_offset']}
        if heb_d not in heb_matches_with_codes:
            heb_matches_with_codes.append(heb_d)
        # heb_matches_with_codes.append((cand + "-" + word_match + " ( " + match_eng + " )" + " : " + str(sim), match_cui))
    return heb_matches_with_codes

def add_eng_umls_data(eng_matches, umls_data, row):
    eng_matches_with_codes = []
    for match_d in eng_matches:
        idx_of_match_in_df = umls_data[umls_data[STRING_COLUMN].apply(lambda x: x.lower() == match_d['umls_match'])].index[0]
        match_cui = umls_data.iloc[idx_of_match_in_df]['CUI']
        match_tui = umls_data.iloc[idx_of_match_in_df]['TUI']
        semantic_type = get_semantic_type(match_tui)
        eng_d = {'cand_match': match_d['cand_match'], 'umls_match': match_d['umls_match'], 'sim': match_d['sim'],
                 'cui': match_cui, 'match_tui': match_tui, 'semantic_type': semantic_type, 'all_match_occ': match_d['all_match_occ'],
                 'curr_occurence_offset': match_d['curr_occurence_offset']}
        if eng_d not in eng_matches_with_codes:
            eng_matches_with_codes.append(eng_d)
    return eng_matches_with_codes

def get_eng_text(msg_tokenized_txt):
    msg_words = msg_tokenized_txt.split(" ")
    msg_eng_text = " ".join([w.lower() for w in msg_words if w != "" and w != " " and len(w) > 2 and word_is_english(w)])
    return msg_eng_text

if __name__ == '__main__':
    main()
