import argparse

import joblib
from deepsegment import DeepSegment
import sys
import os

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, "../../amr/"))

segmenter = DeepSegment('en', tf_serving=False)


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

sentences_with_amr_objects = joblib.load(args.input_path)

for obj in sentences_with_amr_objects:
    obj[1].add_sentence_segments(
        segmenter.segment(obj[0].strip())
    )

joblib.dump(sentences_with_amr_objects, args.output_path)
