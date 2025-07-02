# Rewritten version of convert_questions.py for Python 3.10+
# Replacing `pattern.en` and `nectar.corenlp` with `spaCy`

import json
import random
import re
import argparse
from nltk.corpus import wordnet
from nltk.stem import LancasterStemmer
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
stemmer = LancasterStemmer()

# Mapping for Penn Treebank POS tags to WordNet POS
def penn2morphy(penntag):
    if penntag.startswith('NN'):
        return wordnet.NOUN
    if penntag.startswith('VB'):
        return wordnet.VERB
    if penntag.startswith('JJ'):
        return wordnet.ADJ
    if penntag.startswith('RB'):
        return wordnet.ADV
    return None

def get_pos(w, tag):
    wn_tag = penn2morphy(tag)
    if wn_tag:
        return wn_tag
    return wordnet.NOUN

def tokenize_and_tag(question):
    doc = nlp(question)
    return [(token.text, token.tag_) for token in doc]

def lemmatize(word, tag):
    wn_tag = penn2morphy(tag)
    if wn_tag:
        lemma = wordnet.morphy(word, wn_tag)
        return lemma if lemma else word
    return word

def stem(word):
    return stemmer.stem(word)

def transform_question(question):
    # Tokenize and POS tag the question
    tagged = tokenize_and_tag(question)

    # Rebuild question and track lemmas/stems
    tokens = []
    lemmas = []
    stems = []
    for word, tag in tagged:
        tokens.append(word)
        lemmas.append(lemmatize(word, tag))
        stems.append(stem(word))

    return {
        "original": question,
        "tokens": tokens,
        "lemmas": lemmas,
        "stems": stems,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["spacy"], help="Processing mode")
    parser.add_argument("-d", "--dataset", required=True, help="Path to SQuAD-style JSON dataset")
    args = parser.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)

    out = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                question = qa["question"]
                info = transform_question(question)
                info["id"] = qid
                out.append(info)

    output_path = f"{args.dataset}.processed.json"
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved processed questions to {output_path}")

if __name__ == '__main__':
    main()
