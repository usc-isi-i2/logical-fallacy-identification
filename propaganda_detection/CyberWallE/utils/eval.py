"""
Span identification evaluation.
Used for inspecting the technique-level recall scores.

Adapted from tools/task-SI_scorer.py
(Giovanni Da San Martino, 2019, GPL license 0.1)

"""

import importlib
import inspect
import os
import sys
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
scorer = importlib.import_module('tools.task-SI_scorer')

DEV_GS = '../datasets/dev-task-TC.labels'  # '../data/si-dev-GS.txt'
TEST_GS = '../data/si-test-GS.txt'
PRED_FILE = '../data/post_spans_dev.txt'


def load_annotation_list_from_file(filename, n_cols=3):
    article_idx = 0
    if n_cols == 3:
        technique_idx = None
        start_idx = 1
        end_idx = 2
    elif n_cols == 4:
        technique_idx = 1
        start_idx = 2
        end_idx = 3
    annotations = {}
    with open(filename, "r") as f:
        for row_number, line in enumerate(f.readlines()):
            row = line.rstrip().split("\t")
            if row[article_idx] not in annotations.keys():
                annotations[row[article_idx]] = []
            technique = "propaganda"
            if technique_idx:
                technique = row[technique_idx]
            annotations[row[article_idx]].append([technique,
                                                  set(range(int(row[start_idx]),
                                                            int(row[end_idx])))])
    return annotations


def compute_score_pr(submission_annotations, gold_annotations,
                     technique_names):
    prec_denominator = sum(
        [len(annotations) for annotations in submission_annotations.values()])
    rec_denominator = sum(
        [len(annotations) for annotations in gold_annotations.values()])
    cumulative_Spr_prec = 0
    cumulative_Spr_rec = 0
    if technique_names:
        technique_Spr_rec = {propaganda_technique: 0
                             for propaganda_technique in technique_names}
        technique_rec_denom = {propaganda_technique: 0
                               for propaganda_technique in technique_names}

    for article_id in submission_annotations.keys():
        try:
            gold_data = gold_annotations[article_id]
        except KeyError:
            continue

        for gd in gold_data:
            gd_annotation_length = len(gd[1])
            for sd in submission_annotations[article_id]:
                sd_annotation_length = len(sd[1])
                intersection = len(sd[1].intersection(gd[1]))
                try:
                    Spr_prec = intersection / sd_annotation_length
                except ZeroDivisionError:
                    Spr_prec = 0.0
                cumulative_Spr_prec += Spr_prec

                try:
                    Spr_rec = intersection / gd_annotation_length
                except ZeroDivisionError:
                    Spr_rec = 0.0
                cumulative_Spr_rec += Spr_rec

                if technique_names:
                    technique_Spr_rec[gd[0]] += Spr_rec
            if technique_names:
                technique_rec_denom[gd[0]] += 1

    if technique_names:
        for article_id in set(gold_annotations.keys()) - set(submission_annotations.keys()):
            for gd in gold_annotations[article_id]:
                technique_rec_denom[gd[0]] += 1

    p, r, f1 = scorer.compute_prec_rec_f1(cumulative_Spr_prec,
                                          prec_denominator,
                                          cumulative_Spr_rec,
                                          rec_denominator)
    print('PREC\t{}\tREC\t{}\tF1\t{}\n'.format(p, r, f1))
    if technique_names:
        print('Recall values by technique')
    for t in technique_names:
        try:
            rec = technique_Spr_rec[t] / technique_rec_denom[t]
        except ZeroDivisionError:
            rec = 0
        print('{}\t{:.4f}'.format(t, rec))


def eval(pred_file, gs_file, by_technique=False):
    technique_names = ["propaganda"]
    if by_technique:
        technique_names = ["Name_Calling,Labeling", "Doubt",
                           "Exaggeration,Minimisation", "Repetition",
                           "Appeal_to_fear-prejudice", "Flag-Waving",
                           "Whataboutism,Straw_Men,Red_Herring",
                           "Black-and-White_Fallacy", "Slogans",
                           "Causal_Oversimplification", "Appeal_to_Authority",
                           "Thought-terminating_Cliches", "Loaded_Language",
                           "Bandwagon,Reductio_ad_hitlerum"]
    submission_annotations = load_annotation_list_from_file(pred_file)
    gold_annotations = load_annotation_list_from_file(gs_file, n_cols=4)
    if not scorer.check_annotation_spans(submission_annotations, False):
        print("Error in submission file")
        sys.exit()
    scorer.check_annotation_spans(gold_annotations, True)

    compute_score_pr(submission_annotations, gold_annotations, technique_names)


eval(PRED_FILE, DEV_GS, by_technique=True)
