# MDTEL - Medical-Deep-Transliteration-Entity-Linking

This repository contains the code for the paper "Cross-Lingual UMLS Entity Linking in Online Health Communities" 

As described in the paper, there are 3 main steps: 

1. **Transliteration**: A forward transliteration model that we train and use to transliterate the UMLS into Hebrew.
2. **High Recall Matcher**: A matcher to match spans from the post to the transliterated UMLS, producing a high recall list of candidate
matches.
3. **Contextual Relevance**: A contextual relevance model that filters the high recall list by detecting matches that are not used in a medical
sense and produces a smaller, more relevant list. 

Here is a link to the data directory: pass
When referring to "data", it means the data folder where ever you choose to place it. 

We now describe steps to reproduce each one of the steps.

## Transliteration

The code for this part is here: https://github.com/snukky/news-translit-nmt

This step is composed of: 
1. Train a transliteration model on the training data at "data\transliteration\training_transliteration_model\synthesized_dataset_ready_for_training.txt"
2. Use the trained model in order to predict - transliterate the umls data at "data\transliteration\inference_predict_all_umls\mrconso.txt". 

The output of this step is the transliterated UMLS that we can now use for the next phase. 
The output is at "data\transliteration\inference_predict_all_umls\mrconso_transliterated.txt" 

## High Recall Matcher

Run the code at "src\high_recall_matcher.py".
If DEBUG = True if you want it to perform on 100 posts. 

The output will be at: "data\high_recall_matcher\output"

## Contextual Relevance

This step includes several parts:
1. Relatedness
2. Language model: ULMFiT and n-gram model
3. Match count and match frequency

After preparing all of the feature parts, the code to run will include two files:
1. "src\contextual_relevance\extract_features_for_ml_model.py'
This file takes as input the feature outputs from all of relatedness, language model, and match stats outputs, 
and prepare a training dataset for machine learning contextual relevance model.  

### Language models
The data for this part is at: http://u.cs.biu.ac.il/~yogo/hebwiki/

Training two language model.
First language model: Simple n-gram model. The code for this part is here: https://metacpan.org/pod/UMLS::Similarity

The second language model is based on fastai language model. 

#### Fastai ULMFiT Model

At inference step we added a function to fastai's learner.py, 
LanguageLearner class:
```python
def get_prob_of_word_in_context(self, context: str, word: str):
    self.model.reset()
    xb,yb = self.data.one_item(context)
    res = self.pred_batch(batch=(xb, yb))[0][-1]
    normalized_scores = F.softmax(res, dim=0)
    index_of_word = self.data.vocab.stoi[word]
    prob_of_word_given_context = normalized_scores[index_of_word]
    return prob_of_word_given_context
``` 

This function returns the probability of a word given context. 
For example we expect P(Red | This car is) > P(Frog | This car is) (P = get_prob_of_word_in_context)

The file "src\contextual_relevance\language_models\ulmfit_predict_probs.py" uses the trained model in order 

### Relatedness
The software that is needed in order to use this part:
https://metacpan.org/pod/UMLS::Similarity

After installation, run the following command:

<code>
umls-similarity.pl -user=root -password=admin --infile="eng_words_for_relatedness.txt" --measure=vector > output_relatedness.txt  
</code>

While 'eng_words_for_relatedness' is the terms you want to measure the relatedness, for example:

<code>
diabetes<>shivering
<br>
diabetes<>Terpsichore praeceps
<br>
diabetes<>Left geniohyoid
</code>  
  
The relatedness output for our 3 communities is at "data\contextual_relevance\relatedness" at this format:
<code>
0.6983<>diabetes(C0011849)<>shivering(C0036973)
<br>
0.1726<>diabetes(C0011847)<>Terpsichore praeceps(C2820108)
<br>
0.4248<>diabetes(C0011849)<>Left geniohyoid(C0921041)
</code>

### Match count and match frequency

