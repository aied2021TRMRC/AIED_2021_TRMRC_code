import re
import os
import sys
import random
import string
import logging
import argparse
import json
import torch
import pickle
import spacy
import tqdm
import numpy as np
from os.path import basename, dirname
from datetime import datetime
from collections import Counter, defaultdict
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGenPre, BatchGen
from my_utils.tokenizer import reform_text, Vocabulary, END
from my_utils.log_wrapper import create_logger
from my_utils.word2vec_utils import load_glove_vocab, build_embedding
from my_utils.data_utils import build_data_p

mp = 'checkpoint/checkpoint_last.pt'
my_meta = 'data-squad/meta_v2.pick'
my_covec = 'data-squad/MT-LSTM.pt'
elmo_options_path = 'data-squad/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
elmo_weight_path = 'data-squad/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
glv = 'data/glove.840B.300d.txt'

have_gpu = True
test_file = sys.argv[1]
output_file = sys.argv[2]
batch_size = 16
max_len = 5
repeat_times = 2
avg_on = False

logger = create_logger(__name__, to_disk=False)
workspace = dirname(os.path.realpath(__file__))
# go back to the root
# workspace="/home/work/project/keypoint/san_mrc-master"
model_path = os.path.join(workspace, mp)
glove_path = os.path.join(workspace, glv)
glove_dim = 300
meta_path = os.path.join(workspace, my_meta)
mtlstm_path = os.path.join(workspace, my_covec)
n_threads = 16

pad='-' * 10
logger.info('{}Resource Path{}'.format(pad, pad))
logger.info('workspace:{}'.format(workspace))
logger.info('model path:{}'.format(model_path))
logger.info('test file:{}'.format(test_file))
logger.info('output file:{}'.format(output_file))
logger.info('glove file:{}'.format(glove_path))
logger.info('meta file:{}'.format(meta_path))
logger.info('mtlstm file:{}'.format(mtlstm_path))
logger.info('processing data ...')

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
tr_vocab = meta['vocab']
vocab_tag = meta['vocab_tag']
vocab_ner = meta['vocab_ner']
logger.info('loaded meta data')
print(glove_path)
print(workspace)
glove_vocab = load_glove_vocab(glove_path, glove_dim)
logger.info('loaded glove vector')

# setting up spacy
NLP = spacy.load('en', disable=['vectors', 'textcat', 'parser'])

# def load_data(path, is_train=True):
#     rows = []
#     with open(path, encoding="utf8") as f:
#         data = json.load(f)['data']
#     cnt = 0
#     for article in tqdm.tqdm(data, total=len(data)):
#         for paragraph in article['paragraphs']:
#             cnt += 1
#             context = paragraph['context']
#             context = '{} {}'.format(context, END)
#             for qa in paragraph['qas']:
#                 uid, question = str(qa['id']), qa['question']
#                 is_impossible = qa.get('is_impossible', False)
#                 label = 1 if is_impossible else 0
#                 sample = {'uid': uid, 'context': context, 'question': question, 'label':label}
#                 rows.append(sample)
#     logger.info('loaded {} samples'.format(len(rows)))
#     return rows

def load_data(path, is_train=True, v2_on=True):
    rows = []
    with open(path, encoding="utf8") as f:
        data = json.load(f)['data']
    for article in tqdm.tqdm(data, total=len(data)):
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            if v2_on:
                context = '{} {}'.format(context, END)
            for qa in paragraph['qas']:
                uid, question = qa['id'], qa['question']
                answers = qa.get('answers', [])
                # used for v2.0
                is_impossible = qa.get('is_impossible', False)
                label = 1 if is_impossible else 0
                if is_train:
                    if (v2_on and label < 1 and len(answers) < 1) or ((not v2_on) and len(answers) < 1): continue
                    if len(answers) > 0:
                        answer = answers[0]['text']
                        answer_start = answers[0]['answer_start']
                        answer_end = answer_start + len(answer)
                        if v2_on:
                            sample = {'uid': uid, 'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'answer_end':answer_end, 'label': label}
                        else:
                            sample = {'uid': uid, 'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'answer_end':answer_end}
                    else:
                        answer = END
                        answer_start = len(context) - len(END)
                        answer_end = len(context)
                        sample = {'uid': uid, 'context': context, 'question': question, 'answer': answer, 'answer_start': answer_start, 'answer_end':answer_end, 'label': label}
                else:
                    sample = {'uid': uid, 'context': context, 'question': question, 'answer': answers, 'answer_start': -1, 'answer_end':-1}
                rows.append(sample)
                # if DEBUG_ON and (not is_train) and len(rows) == DEBUG_SIZE:
                #     return rows
    return rows


def build_vocab(data, tr_vocab, n_threads=16):
    nlp = spacy.load('en', disable=['vectors', 'textcat', 'parser', 'tagger', 'ner'])
    text = [reform_text(sample['context']) for sample in data] + [reform_text(sample['question']) for sample in data]
    parsed = nlp.pipe(text, batch_size=10000, n_threads=n_threads)
    tokens = [w.text for doc in parsed for w in doc if len(w.text) > 0]
    new_vocab = list(set([w for w in tokens if w not in tr_vocab and w in glove_vocab]))
    for w in new_vocab:
        tr_vocab.add(w)
    return tr_vocab

# load data
data = load_data(test_file)
# load model
checkpoint = torch.load(model_path)
opt = checkpoint['config']
opt['covec_path'] = mtlstm_path
opt['cuda'] = have_gpu
opt['elmo_options_path'] = elmo_options_path
opt['elmo_weight_path'] = elmo_weight_path
opt['multi_gpu'] = False
opt['max_len'] = max_len
num_tune = opt['tune_partial'] if not opt['fix_embeddings'] else 0
threshold = opt['classifier_threshold']

# set seeds
# torch.random.set_rng_state(checkpoint['torch_state'])
# torch.cuda.random.set_rng_state(checkpoint['torch_cuda_state'])

# build vocab
tr_vocab = Vocabulary.build(tr_vocab.get_vocab_list()[:num_tune])

# vocab
test_vocab = build_vocab(data, tr_vocab, n_threads=n_threads)
logger.info('Collected vocab')
test_embedding = build_embedding(glove_path, test_vocab, glove_dim)
logger.info('Got embedding')

data = build_data_p(data, test_vocab, vocab_tag, vocab_ner, NLP, thread=n_threads, )
batches =  BatchGenPre(data, batch_size, have_gpu, is_train=False)
state_dict = checkpoint['state_dict']
model = DocReaderModel(opt, state_dict = state_dict)
logger.info('Loaded model!')

model.setup_eval_embed(torch.Tensor(test_embedding))
if have_gpu:
    model.cuda()


predictions = {}
idx = 0
batches.reset()
for bid, batch in enumerate(batches):
    if idx % 100 == 0:
        logger.info('predicting {}-th ...'.format(idx))
    idx += 1
    phrase, _, scores = model.predict(batch)

    uids = batch['uids']
    for uid, pred, score in zip(uids, phrase, scores):
        if score > threshold:
            pred = ''
        predictions[uid] = pred

with open(output_file, 'w') as f:
    json.dump(predictions, f)