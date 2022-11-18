import argparse
import re
import joblib
from deepsegment import DeepSegment
import sys
import os

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../../amr/"))

import os
import sys

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../../../cbr_analyser/amr/"))


def clean_text(sent: str) -> str:
    sent = sent.strip()
    sent = sent.replace('-', ' ')
    sent = sent.replace('_', ' ')
    sent = sent.replace('  ', ' ')
    sent = sent.replace('\n', '. ')
    sent = sent.replace('"', '')
    sent = sent.replace("'", '')
    sent = sent.replace('..', '.')
    return sent


segmenter = DeepSegment('en', tf_serving=False)


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

sentences_with_amr_objects = joblib.load(args.input_path)

for obj in sentences_with_amr_objects:
    sent = clean_text(obj[0])

    segments = segmenter.segment(sent)
    final_results = []
    for segment in segments:
        final_results.extend(
            re.split('[.!?]', segment)
        )
    final_results = [x for x in final_results if (
        x != '') and not str.isspace(x)]
    obj[1].add_sentence_segments(
        final_results
    )

joblib.dump(sentences_with_amr_objects, args.output_path)
