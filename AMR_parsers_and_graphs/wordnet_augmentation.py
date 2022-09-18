from collections import defaultdict
from nltk.corpus import wordnet as wn
from consts import PATH_TO_MASKED_SENTENCES_AMRS_WITH_LABEL2WORD, PATH_TO_MASKED_SENTENCES_AMRS_WITH_LABEL2WORD_WORDNET
import joblib
from tqdm import tqdm
import numpy as np
from IPython import embed
from custom_logger import get_logger
import os


logger = get_logger(f"{__name__}.{os.path.basename(__file__)}")


def get_word_meronyms(word: str):
    all_meronyms = set()
    for syn in wn.synsets(word):
        for meronym in syn.part_meronyms():
            for lemma in meronym.lemmas():
                all_meronyms.add(lemma.name())
        for meronym in syn.substance_meronyms():
            for lemma in meronym.lemmas():
                all_meronyms.add(lemma.name())
    return [word.lower().replace('_', ' ') for word in all_meronyms]


def get_word_holonyms(word: str):
    all_holonyms = set()
    for syn in wn.synsets(word):
        for meronym in syn.part_holonyms():
            for lemma in meronym.lemmas():
                all_holonyms.add(lemma.name())
        for meronym in syn.substance_holonyms():
            for lemma in meronym.lemmas():
                all_holonyms.add(lemma.name())
    return [word.lower().replace('_', ' ') for word in all_holonyms]


def check_if_words_have_meronym_relationship(word1: str, word2: str):
    word1_meronyms = get_word_meronyms(word1)
    word2_meronyms = get_word_meronyms(word2)

    if word1 in word2_meronyms:
        return 0
    elif word2 in word1_meronyms:
        return 1
    return -1


def check_if_words_have_holonyms_relationship(word1: str, word2: str):
    word1_holonyms = get_word_holonyms(word1)
    word2_holonyms = get_word_holonyms(word2)

    if word1 in word2_holonyms:
        return 0
    elif word2 in word1_holonyms:
        return 1
    return -1


def find_synsets(word: str):
    synonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return [word.lower().replace('_', ' ') for word in set(synonyms)]


def get_entailment_graph(verb):
    all_entailments = set()
    for syn in wn.synsets(verb):
        for entailed_verb in syn.entailments():
            for lemma in entailed_verb.lemmas():
                all_entailments.add(lemma.name())
    return [word.lower().replace('_', ' ') for word in all_entailments]

def check_if_two_verbs_have_entailment_relations(verb1: str, verb2: str):
    verb1_entailed = get_entailment_graph(verb1)
    verb2_entailed = get_entailment_graph(verb2)

    if verb1 in verb2_entailed:
        return 0
    elif verb2 in verb1_entailed:
        return 1
    return -1

def find_antonyms(word: str):
    antonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonyms.append(ant.name())
    return [word.lower().replace('_', ' ') for word in set(antonyms)]

def check_if_words_are_antonyms(word1: str, word2: str):
    word1_ants = find_antonyms(word1)
    word2_ants = find_antonyms(word2)
    return word1 in word2_ants or word2 in word1_ants

def check_if_words_are_antonyms(word1: str, word2: str):
    word1_ants = find_antonyms(word1)
    word2_ants = find_antonyms(word2)
    return word1 in word2_ants or word2 in word1_ants


def check_if_words_are_synonyms(word1: str, word2: str):
    word1_syns = find_synsets(word1)
    word2_syns = find_synsets(word2)
    return word1 in word2_syns or word2 in word1_syns


def get_hypernyms_to_root(word: str):
    hypernyms = []
    for syn in wn.synsets(word):
        for entity in syn.hypernym_paths()[0]:
            for lemma in entity.lemmas():
                hypernyms.append(lemma.name())
    return set(hypernyms)

def get_hypernyms_in_width_and_depth(word: str, level: int = 0):
    all_hypernyms = []
    hypernyms_in_queue = [(word, 0)]
    processed_nodes = []
    while len(hypernyms_in_queue):
        word, idx = hypernyms_in_queue.pop(0)
        processed_nodes.append(word)
        if idx == level:
            break
        for syn in wn.synsets(word):
            for hypernyms in syn.hypernyms():
                for lemma in hypernyms.lemmas():
                    all_hypernyms.append(lemma.name())
                    if lemma.name() not in processed_nodes:
                        hypernyms_in_queue.append((lemma.name(), idx + 1))
    return set(all_hypernyms)

def get_all_hypernyms(word: str, level_in_depth = 0):
    all_hypernyms = set.union(
        get_hypernyms_in_width_and_depth(word = word, level=level_in_depth),
        get_hypernyms_to_root(word = word)
    )
    return [word.lower().replace('_', ' ') for word in all_hypernyms]


def check_if_words_have_parent_childre_relation(word1: str, word2: str):
    word1_hypernyms = get_all_hypernyms(word1, level_in_depth=0)
    word2_hypernyms = get_all_hypernyms(word2, level_in_depth=0)

    if word1 in word2_hypernyms:
        return 0
    elif word2 in word1_hypernyms:
        return 1
    return -1
    

if __name__ == "__main__":

    statistics = defaultdict(list)

    sentences_with_amr_container = joblib.load(PATH_TO_MASKED_SENTENCES_AMRS_WITH_LABEL2WORD)
    for obj in tqdm(sentences_with_amr_container, leave = False):

        syns = 0
        ants = 0
        parents = 0
        entails = 0
        meronyms = 0
        holonyms = 0

        graph = obj[1].graph_nx
        label2word = obj[1].label2word
        edges = graph.edges(data = False)
        nodes = graph.nodes(data = False)


        for node1 in nodes:
            for node2 in nodes:
                node1_label = label2word[node1].lower()
                node2_label = label2word[node2].lower()
                if node1_label != node2_label and (node1, node2) not in edges and (node2, node1) not in edges:
                    
                    
                    if check_if_words_are_synonyms(node1_label, node2_label):
                        graph.add_edge(node1, node2, label = 'syn')
                        graph.add_edge(node2, node1, label = 'syn')
                        syns += 1
                        

                    if check_if_words_are_antonyms(node1_label, node2_label):
                        graph.add_edge(node1, node2, label = 'ant')
                        graph.add_edge(node2, node1, label = 'ant')
                        ants += 1


                    entailment_relationship = check_if_two_verbs_have_entailment_relations(node1_label, node2_label)
                    if entailment_relationship == 0:
                        graph.add_edge(node2, node1, label = 'entails')
                        entails += 1
                    elif entailment_relationship == 1:
                        graph.add_edge(node1, node2, label = 'entails')
                        entails += 1


                    meronym_relationship = check_if_words_have_meronym_relationship(node1_label, node2_label)
                    if meronym_relationship == 1:
                        graph.add_edge(node2, node1, label = 'part_of')
                        meronyms += 1

                    elif meronym_relationship == 0:
                        graph.add_edge(node1, node2, label = 'part_of')
                        meronyms += 1
                    

                    holonym_relationship = check_if_words_have_holonyms_relationship(node1_label, node2_label)
                    if holonym_relationship == 1:
                        graph.add_edge(node1, node2, label = 'part_of')
                        meronyms += 1

                    elif holonym_relationship == 0:
                        graph.add_edge(node2, node1, label = 'part_of')
                        meronyms += 1


        for edge_type in ["syn", "ant", "entail", 'part_of']: 
            logger.info(
                f"{edge_type}: {[(label2word[edge[0]], label2word[edge[1]]) for edge in graph.edges(data = True) if edge[2]['label'] == edge_type]}"
            )
        logger.info(f"{20 * '#'}\n")
        statistics['syns'].append(syns)
        statistics['ants'].append(ants)
        statistics['entails'].append(entails)
        statistics['meronyms'].append(meronyms)
    
    print(f"Number of edges added based on words being synonym: {np.mean(statistics['syns'])}")
    print(f"Number of edges added based on words being antonyms: {np.mean(statistics['ants'])}")
    print(f"Number of edges added based on words being entailed: {np.mean(statistics['entails'])}")
    print(f"Number of edges added based on words being part_of: {np.mean(statistics['meronyms'])}")
    # embed()
    joblib.dump(
        sentences_with_amr_container,
        PATH_TO_MASKED_SENTENCES_AMRS_WITH_LABEL2WORD_WORDNET
    )


        
                    




