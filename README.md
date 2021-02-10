# AIED_2021_TRMRC_code

In this repo, we publish our code of experiments in paper: Automatic Task Requirements Writing Evaluation With Feedback via Machine Reading Comprehension

We explored many advanced MRC methods, such as BERT, BART, RoBERTa and SAN etc. , to address automatic response locating problem in task requirement questions.

## requirements 

 
 We only test our code on **python 3.6.8**.

 - Transformers

 We use version 3.3.1 of transformers.
  https://github.com/huggingface/transformers 

```
 pip install transformers==3.3.1
```

- Spacy

We use version 2.2.4 of spacy and its small english model `en-core-web-sm`.

```
 pip install spacy==2.2.4 
 python -m spacy download en 
```

## DataSet 

- SQuAD 2.0 
   
   Second version of Stanford Question Answering Dataset. 
     [ https://www.aclweb. org/anthology/P18-2124]
- SED
  
   Student Essay Dataset

- SQuAD 2.0 & SED 

   Dataset consists of SED and SQuAD 2.0

## Training and Evaluation


### Train BERT, BART, RoBERTa, ALBERT

Run `train.sh` in directories named by each models.
Taking training BERT as an example:

```
sh ./bert/train.sh
```
After training, a fine-tuned model could be found at `./bert/models/`.

### Evaluating: BERT, BART, RoBERTa, ALBERT 

 Taking BERT as an example:

```shell
 python ./bert/bert_performance_test.py -f ./data/SED/test.json
```

### Train and evaluating SAN and Ours

   See README.md in `./SAN` and `./ours`.









