import difflib

from fastai.text import *
import sys

module_path = os.path.abspath(os.path.join('..', '..', '..', '..', os.getcwd()))
sys.path.append(module_path)

from config import data_dir, DEBUG

input_dir = data_dir + r"contextual_relevance"
initialized_trainig_dataset_dir = data_dir + r"contextual_relevance\initialized_training_dataset"
output_dir = data_dir + r"contextual_relevance\language_models\output"

def get_ulmfit_feats(row, loaded_learn):
    match = row['match']
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
        p_2_window = ngram_model[context][row['match']]
    return p_2_window



def handle_community(community, ulmfit_model, ngram_model):
    print(f"community: {community}")
    df = pd.read_excel(initialized_trainig_dataset_dir + os.sep + community + ".xlsx")

    preds_10_window = []
    preds_6_window = []
    preds_3_window = []
    preds_2_window = []

    for row_idx, row in df.iterrows():
        if row_idx % 10 == 0:
            print(f"row_idx: {row_idx}, df len: {len(df)}")
        p_2_window = get_ngram_feats(row, ngram_model)

        # if p_2_window > 0:
        #     print(f"{row['match_3_window'], p_2_window}")
        p_10_window, p_6_window, p_3_window = get_ulmfit_feats(row, ulmfit_model)

        preds_2_window.append(p_2_window)  # Ngram

        preds_10_window.append(p_10_window)  # ULMFit
        preds_6_window.append(p_6_window)
        preds_3_window.append(p_3_window)

    df['pred_10_window'] = preds_10_window
    df['pred_6_window'] = preds_6_window
    df['pred_3_window'] = preds_3_window
    df['pred_2_window'] = preds_2_window
    cols = list(df.columns)
    print(f"cols: {cols}, {len(cols)}")
    fpath = output_dir + os.sep + community + '_output.xlsx'
    print(f"Writing file at shape: {df.shape} to fpath: {fpath}")
    df.to_excel(fpath, index=False)

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
    print(dict(two_gram_model[('מתמטיים', 'חדשים')]))
    ngram_model = two_gram_model
    print(f"Number of grams: {len(ngram_model.keys())}")
    return ngram_model


def get_ulmfit_model():
    ulmfit_model = load_learner(input_dir + os.sep + "language_models", 'ulmfit_lm.pickle')
    TEXT = "במהלך השנה 1948 קמה מדינת ישראל"
    N_WORDS = 40
    N_SENTENCES = 1
    print("\n".join(ulmfit_model.predict(TEXT, N_WORDS, temperature=0.9) for _ in range(N_SENTENCES)))
    return ulmfit_model


if __name__ == '__main__':
    ulmfit_model, ngram_model = get_language_models()
    handle_community('diabetes', ulmfit_model, ngram_model)
    # handle_community('sclerosis', ulmfit_model, ngram_model)
    # handle_community('depression', ulmfit_model, ngram_model)

    print("Done")