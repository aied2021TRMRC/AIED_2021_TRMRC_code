# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
# import sys
# print(sys.path)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate_official2 import eval_squad

def get_score1(args):
    cof = [1, 1]
    best_cof = [1]
    all_scores = collections.OrderedDict()
    idx = 0
    for input_file in args.input_null_files:
        with open(input_file, 'r') as reader:
            input_data = json.load(reader, strict=False)
            for (key, score) in input_data.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(cof[idx] * score)
        idx += 1
    output_scores = {}
    for (key, scores) in all_scores.items():
        mean_score = 0.0
        for score in scores:
            mean_score += score
        mean_score /= float(len(scores))
        output_scores[key] = mean_score

    idx = 0
    all_nbest = collections.OrderedDict()
    for input_file in args.input_nbest_files:
        with open(input_file, "r") as reader:
            input_data = json.load(reader, strict=False)
            for (key, entries) in input_data.items():
                if key not in all_nbest:
                    all_nbest[key] = collections.defaultdict(float)
                for entry in entries:
                    all_nbest[key][entry["text"]] += best_cof[idx] * entry["probability"]
        idx += 1
    output_predictions = {}
    for (key, entry_map) in all_nbest.items():
        sorted_texts = sorted(
            entry_map.keys(), key=lambda x: entry_map[x], reverse=True)
        best_text = sorted_texts[0]
        output_predictions[key] = best_text

    best_th = args.thresh

    for qid in output_predictions.keys():
        if output_scores[qid] > best_th:
            output_predictions[qid] = ""

    model_dir = args.model_dir
    output_prediction_file = os.path.join(model_dir, "self_test_predictions.json")
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(output_predictions, indent=4) + "\n")

    results = eval_squad(args.predict_file, output_prediction_file, None, 0.0)
    print(results)
    with open(os.path.join(model_dir, "self_test_result.txt"), "a") as writer:
        for key in sorted(results.keys()):
            writer.write("%s = %s\t" % (key, str(results[key])))
            writer.write("\t\n")


def main():
    parser = argparse.ArgumentParser()

    model_dir = '/share/jpl/sentence_paraphrasing/AwesomeMRC/transformer-mrc/models/squad/'
    av_dir = model_dir + 'av_no_squad_self_electra-large-v2'
    cls_dir = model_dir + 'cls_no_squad_self_electra-large-v2'
    input_null_files = [
        cls_dir + '/self_test_cls_score.json',
        av_dir + '/null_odds_self_test.json'
    ]
    input_nbest_files = [
        av_dir + '/nbest_predictions_self_test.json'
    ]
    # predict_file = '/share/jpl/sentence_paraphrasing/AwesomeMRC/transformer-mrc/squad-2.0/dev-v2.0.json'
    predict_file = '/share/jpl/sentence_paraphrasing/AwesomeMRC/transformer-mrc/self_data/11_05_squad/test.json'
    # Required parameters
    parser.add_argument('--input_null_files', type=list, default=input_null_files)
    parser.add_argument('--input_nbest_files', type=list, default=input_nbest_files)
    parser.add_argument('--thresh', default=0, type=float)
    parser.add_argument("--predict_file", default=predict_file)
    parser.add_argument("--model_dir", default=model_dir)
    args = parser.parse_args()
    get_score1(args)

if __name__ == "__main__":
    main()
