import argparse
import difflib

from fastai.text import *
import sys

module_path = os.path.abspath(os.path.join('..', '..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir

input_dir = data_dir + r"contextual_relevance"
initialized_trainig_dataset_dir = data_dir + r"contextual_relevance\initialized_training_dataset"
output_dir = data_dir + r"contextual_relevance\language_models\output"

def get_ulmfit_feats(row, loaded_learn):
    match = row['cand_match']
    match_10_window = row['match_10_window']
    match_6_window = row['match_6_window']
    match_3_window = row['match_3_window']

    p_10_window = loaded_learn.get_prob_of_word_in_context(match_10_window, match).item()
    p_6_window = loaded_learn.get_prob_of_word_in_context(match_6_window, match).item()
    p_3_window = loaded_learn.get_prob_of_word_in_context(match_3_window, match).item()

    return p_10_window, p_6_window, p_3_window


def get_ngram_feats(row, ngram_model, n=2):
    if str(row['match_3_window']) == 'nan':
        p_2_window = 0
    else:
        context = tuple(row['match_3_window'].split(" ")[3 - n:])
        p_2_window = ngram_model[context][row['cand_match']]
    return p_2_window



def handle_community(community, ulmfit_model, ngram_model, model):
    print(f"Language model extractor: community: {community}")
    df = pd.read_csv(initialized_trainig_dataset_dir + os.sep + community + ".csv")
    fpath = output_dir + os.sep + community + '_output.csv'
    cache_df = pd.read_csv(fpath)

    preds_10_window = []
    preds_6_window = []
    preds_3_window = []
    preds_2_window = []

    items_with_cache = 0
    items_without_cache = 0

    for row_idx, row in df.iterrows():
        p_10_window, p_6_window, p_3_window, p_2_window, items_with_cache = get_from_cache(cache_df, items_with_cache, row)

        if (p_10_window, p_6_window, p_3_window, p_2_window) == (None, None, None, None):
            items_without_cache += 1
            if model:
                p_2_window = get_ngram_feats(row, ngram_model)
                p_10_window, p_6_window, p_3_window = get_ulmfit_feats(row, ulmfit_model)
            else:
                p_10_window = None
                p_6_window = None
                p_3_window = None
                p_2_window = None


        preds_2_window.append(p_2_window)  # Ngram

        preds_10_window.append(p_10_window)  # ULMFit
        preds_6_window.append(p_6_window)
        preds_3_window.append(p_3_window)

    df['pred_10_window'] = preds_10_window
    df['pred_6_window'] = preds_6_window
    df['pred_3_window'] = preds_3_window
    df['pred_2_window'] = preds_2_window

    print(f"items_with_cache: {items_with_cache}, items_without_cache: {items_without_cache} ({round((items_without_cache / (items_without_cache + items_with_cache))*100, 1)}%)")

    if model:
        df.to_csv(fpath, index=False, encoding='utf-8-sig')


def get_from_cache(cache_df, items_with_cache, row):
    p_10_window, p_6_window, p_3_window, p_2_window = None, None, None, None
    file_name_df = cache_df[cache_df['file_name'] == row['file_name']]
    match_in_cand_match = file_name_df[file_name_df['cand_match'] == row['cand_match']]
    if len(match_in_cand_match) == 1:
        relevant_row = match_in_cand_match.iloc[0]
    else:
        match_in_10_window = match_in_cand_match[match_in_cand_match['match_10_window'] == row['match_10_window']]
        if len(match_in_10_window) == 1 and match_in_10_window.iloc[0]['curr_occurence_offset'] == row['curr_occurence_offset']:
                relevant_row = match_in_10_window.iloc[0]
        else:
            return p_10_window, p_6_window, p_3_window, p_2_window, items_with_cache

    items_with_cache += 1
    p_10_window = relevant_row['pred_10_window']
    p_6_window = relevant_row['pred_6_window']
    p_3_window = relevant_row['pred_3_window']
    p_2_window = relevant_row['pred_2_window']
    return p_10_window, p_6_window, p_3_window, p_2_window, items_with_cache


def dd2():
    return 0

def dd():
    return defaultdict(dd2)

def get_language_models():
    ulmfit_model = get_ulmfit_model()

    ngram_model = get_ngram_model()

    return ulmfit_model, ngram_model


def get_ngram_model():
    ngram_model_path = input_dir + os.sep + "language_models" + os.sep + 'ngram_models_two_gram.pickle'
    with open(ngram_model_path, 'rb') as f:
        ngram_model = pickle.load(f)
    two_gram_model = ngram_model['two_gram_model']
    # print(dict(two_gram_model[('מתמטיים', 'חדשים')]))
    ngram_model = two_gram_model
    # print(f"Number of grams: {len(ngram_model.keys())}")
    return ngram_model


def get_ulmfit_model():
    ulmfit_model = load_learner(input_dir + os.sep + "language_models", 'ulmfit_lm.pickle')
    TEXT = "במהלך השנה 1948 קמה מדינת ישראל"
    N_WORDS = 40
    N_SENTENCES = 1
    # print("\n".join(ulmfit_model.predict(TEXT, N_WORDS, temperature=0.9) for _ in range(N_SENTENCES)))
    return ulmfit_model


if __name__ == '__main__':
    model = False
    print(f"Model: {model}")

    if model:
        ulmfit_model, ngram_model = get_language_models()
    else:
        ulmfit_model, ngram_model = None, None

    handle_community('sclerosis', ulmfit_model, ngram_model, model)
    handle_community('diabetes', ulmfit_model, ngram_model, model)
    handle_community('depression', ulmfit_model, ngram_model, model)
    print("Language model extractor - Done.")
