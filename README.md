
# Modeling-Label-Semantics-for-Predicting-Motivation

Code for the project - Modeling Label Semantics for Predicting Motivation by Steven Secreti. This work explores the use different ways of utilizing label semantics techniques for improving multi-label motive prediction in short commonsense stories. This project was inspired by the work done in the paper Modeling Label Semantics for Predicting Emotional Reactions -- https://arxiv.org/pdf/2006.05489.pdf. 

## Original Source for Codebase
The codebase for this project was initially copied directly from the codebase used in the above mentioned paper, Modeling Label Semantics for Predicting Emotional Reactions. The original codebase for that paper can be found here: https://github.com/StonyBrookNLP/emotion-label-semantics/tree/master 


## Modified Files & Functions
The following is a list of modified files and functions within those files that I've modified for the sake of this project. Some of the files are entirely new, but inspired by files from the original codebase. In cases like these, I will list the name of the file from the original codebase that they were inspired from.

### Root Directory - Driver Files:
- `maslow_main_bert_base.py`
    - Main driver file for the Maslow's Needs task using the pretrained BERT model on the dataset containing all character-line pairs
    - Inspired by the file `main_bert_base.py` in the original codebase
- `maslow_main_leam_bert_label.py`
    - Main driver file for the Maslow's Needs task using the LEAM architecture on the dataset containing all character-line pairs
    - Inspired by the file `main_leam_bert_label.py` in the original codebase
- `maslow_main_bert_label.py`
    - Main driver file for the Maslow's Needs task using the Label Sentences technique on the dataset containing all character-line pairs
    - Inspired by the file `main_bert_label.py` in the original codebase
- `maslow_action_bert_base.py`
    - Main driver file for the Maslow's Needs task using the pretrained BERT model on the dataset containing only the character-line pairs in which an action occurred
    - Inspired by the file `main_bert_label.py` in the original codebase
- `maslow_action_main_leam_bert_label.py`
    - Main driver file for the Maslow's Needs task using the LEAM architecture on the dataset containing only the character-line pairs in which an action occurred
    - Inspired by the file `main_leam_bert_label.py` in the original codebase
- `maslow_action_main_bert_label.py`
    - Main driver file for the Maslow's Needs task using the Label Sentences technique on the dataset containing only the character-line pairs in which an action occurred
    - Inspired by the file `main_bert_label.py` in the original codebase
- `reiss_main_bert_base.py`
    - Main driver file for the Reiss' Motives task using the pretrained BERT model on the dataset containing all character-line pairs
    - Inspired by the file `main_bert_base.py` in the original codebase
- `reiss_main_leam_bert_label.py`
    - Main driver file for the Reiss' Motives task using the LEAM architecture on the dataset containing all character-line pairs
    - Inspired by the file `main_leam_bert_label.py` in the original codebase
- `reiss_main_bert_label.py`
    - Main driver file for the Reiss' Motives task using the Label Sentences technique on the dataset containing all character-line pairs
    - Inspired by the file `main_bert_label.py` in the original codebase
- `reiss_action_main_bert_base.py`
    - Main driver file for the Reiss' Motives task using the pretrained BERT model on the dataset containing only the character-line pairs in which an action occurred
    - Inspired by the file `main_bert_base.py` in the original codebase
- `reiss_action_main_leam_bert_label.py`
    - Main driver file for the Reiss' Motives task using the LEAM architecture on the dataset containing only the character-line pairs in which an action occurred
    - Inspired by the file `main_leam_bert_label.py` in the original codebase
- `reiss_action_main_bert_label.py`
    - Main driver file for the Reiss' Motives task using the Label Sentences technique on the dataset containing only the character-line pairs in which an action occurred
    - Inspired by the file `main_bert_label.py` in the original codebase

### pybert Directory
#### configs Directory
Configuration files used to tell the driver files the location of the datasets and the checkpoint directories for the given experiment.
- `maslow_basic_config.py`
    - Inspired by the file `basic_config.py` in the original codebase
- `maslow_basic_config_leam2.py`
    - Inspired by the file `basic_config_leam2.py` in the original codebase
- `maslow_basic_config_label_finetune.py`
    - Inspired by the file `basic_config_label_finetune.py` in the original codebase
- `maslow_basic_config_action.py`
    - Inspired by the file `basic_config.py` in the original codebase
- `maslow_basic_config_leam2_action.py`
    - Inspired by the file `basic_config_leam2.py` in the original codebase
- `maslow_basic_config_label_finetune_action.py`
    - Inspired by the file `basic_config_label_finetune.py` in the original codebase
- `reiss_basic_config.py`
    - Inspired by the file `basic_config.py` in the original codebase
- `reiss_basic_config_leam2.py`
    - Inspired by the file `basic_config_leam2.py` in the original codebase
- `reiss_basic_config_label_finetune.py`
    - Inspired by the file `basic_config_label_finetune.py` in the original codebase
- `reiss_basic_config_action.py`
    - Inspired by the file `basic_config.py` in the original codebase
- `reiss_basic_config_leam2_action.py`
    - Inspired by the file `basic_config_leam2.py` in the original codebase
- `reiss_basic_config_label_finetune_action.py`
    - Inspired by the file `basic_config_label_finetune.py` in the original codebase

#### dataset Directory
Created basic checkpoint directories to store model/data checkpoints for the experiments in this project. Also stored the datasets used for this project. The format of the following datasets follow the format of the datasets established and used by Gaonkar et al. (2020) in the above mentioned codebase.
- `train_maslow.csv`
    - Training Data for the Maslow's Needs task containing all character-line pairs.
- `test_maslow.csv`
    - Testing Data for the Maslow's Needs task containing all character-line pairs.
- `train_maslow_all_action.csv`
    - Training Data for the Maslow's Needs task containing only character-line pairs in which an action occurred.
- `test_maslow_all_action.csv`
    - Testing Data for the Maslow's Needs task containing only character-line pairs in which an action occurred.
- `train_reiss.csv`
    - Training Data for the Reiss' Motives task containing all character-line pairs.
- `test_reiss.csv`
    - Testing Data for the Reiss' Motives task containing all character-line pairs.
- `train_reiss_all_action.csv`
    - Training Data for the Reiss' Motives task containing only character-line pairs in which an action occurred.
- `test_reiss_all_action.csv`   
    - Testing Data for the Reiss' Motives task containing only character-line pairs in which an action occurred.

#### io Directory
- `maslow_bert_processor_label_attention2.py`
    - Inspired by `bert_processor_label_attention2.py` in the original codebase
    - Modified the `create_examples` function to create label sentences of the form: `[char] needs [label]`
    - Modified the `get_labels` function to return the 5 Maslow's Needs labels for the Maslow task.
- `maslow_bert_processor_label2.py`
    - Inspired by `bert_processor_label2.py` in the original codebase
    - Modified the `create_examples` function to create label sentences of the form: `[char] needs [label]`
    - Modified the `get_labels` function to return the 5 Maslow's Needs labels for the Maslow task.
- `maslow_task_data_label`
    - Inspired by `task_data_label.py` in the original codebase
    - Modified the `read_data` function to fit the Maslow Labels
- `reiss_bert_processor_label_attention2.py`
    - Inspired by `bert_processor_label_attention2.py` in the original codebase
    - Modified the `create_examples` function to create label sentences of the form: `[char] is motivated by [label]`
    - Modified the `get_labels` function to return the 19 motive labels for the Reiss task.
- `reiss_bert_processor_label2.py`
    - Inspired by `bert_processor_label2.py` in the original codebase
    - Modified the `create_examples` function to create label sentences of the form: `[char] is motivated by [label]`
    - Modified the `get_labels` function to return the 19 motive labels for the Reiss task.
- `reiss_task_data_label`
    - Inspired by `task_data_label.py` in the original codebase
    - Modified the `read_data` function to fit the Reiss Labels

#### model/nn Directory
- `bert_for_multi_label_attention2_no_pretrain.py` 
    - Generalized the `BertForMultiLable` class to accept a dynamic number of output class labels dependent upon the task at hand

#### output Directory
Mainly just created checkpoint folders here

#### test Directory
Created all my own predictor files from scratch because they were missing from the original codebase. Essentially mapped out the process that was occuring in the `validate` method in the training files. Wrote the code to calculate and report the micro-averaged precision, recall, and f1 scores. The predictor files I created are as follows:
- `predictor.py`
- `predictor_label_attention1.py`
- `maslow_predictor.py`
- `maslow_predictor_label_attention1.py`
- `reiss_predictor.py`
- `reiss_predictor_label_attention1.py`

#### train Directory
- `maslow_trainer_old.py`
    - Inspired by `trainer_old.py` that was present in the original codebase.
    - Switched the training and validation code to handle 5 output labels for the Maslow Needs prediction task instead of 8
    - Wrote the code to calculate and report the micro-averaged precision, recall, and f1-scores at the end of every training and validation epoch
- `maslow_trainer_label_attn1.py`
    - Inspired by `trainer_label_attn1.py` that was present in the original codebase.
    - Switched the training and validation code to handle 5 output labels for the Maslow Needs prediction task instead of 8
    - Wrote the code to calculate and report the micro-averaged precision, recall, and f1-scores at the end of every training and validation epoch
- `reiss_trainer_old.py`
    - Inspired by `trainer_old.py` that was present in the original codebase.
    - Switched the training and validation code to handle 19 output labels for the Reiss' Motives prediction task instead of 8
    - Wrote the code to calculate and report the micro-averaged precision, recall, and f1-scores at the end of every training and validation epoch
- `reiss_trainer_label_attn1.py`
    - Inspired by `trainer_label_attn1.py` that was present in the original codebase.
    - Switched the training and validation code to handle 19 output labels for the Reiss' Motives prediction task instead of 8
    - Wrote the code to calculate and report the micro-averaged precision, recall, and f1-scores at the end of every training and validation epoch


## Running experiments
This section covers running each of the experiments that were described in the project report. The model names and corresponding main file names are mentioned below, followed by a list of commands to process the data, run the training and then perform testing:
1. Maslow BERT - `maslow_main_bert_base.py`
    - Do Data: `python maslow_main_bert_base.py --do_data`
    - Training: `python maslow_main_bert_base.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 16`
    - Testing: `python maslow_main_bert_base.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
2. Maslow BERT + LEAM - `maslow_main_leam_bert_label.py`
    - Do Data: `python maslow_main_leam_bert_label.py --do_data`
    - Training: `python maslow_main_leam_bert_label.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 16`
    - Testing: `python maslow_main_leam_bert_label.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
3. Maslow BERT + Label Sentences - `maslow_main_bert_label.py`
    - Do Data: `python maslow_main_bert_label.py --do_data`
    - Training: `python maslow_main_bert_label.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 16`
    - Testing: `python maslow_main_bert_label.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
4. Maslow Action Char-Line Pairs BERT - `maslow_action_bert_base.py`
    - Do Data: `python maslow_action_bert_base.py --do_data`
    - Training: `python maslow_action_bert_base.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 16`
    - Testing: `python maslow_action_bert_base.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
5. Maslow Action BERT + LEAM - `maslow_action_main_leam_bert_label.py`
    - Do Data: `python maslow_action_main_leam_bert_label.py --do_data`
    - Training: `python maslow_action_main_leam_bert_label.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 16`
    - Testing: `python maslow_action_main_leam_bert_label.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
6. Maslow Action BERT + Label Sentences - `maslow_action_main_bert_label.py`
    - Do Data: `python maslow_action_main_bert_label.py --do_data`
    - Training: `python maslow_action_main_bert_label.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 16`
    - Testing: `python maslow_action_main_bert_label.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
7. Reiss BERT - `reiss_main_bert_base.py`
    - Do Data: `python reiss_main_bert_base.py --do_data`
    - Training: `python reiss_main_bert_base.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 8`
    - Testing: `python reiss_main_bert_base.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
8. Reiss BERT + LEAM - `reiss_main_leam_bert_label.py`
    - Do Data: `python reiss_main_leam_bert_label.py --do_data`
    - Training: `python reiss_main_leam_bert_label.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 8`
    - Testing: `python reiss_main_leam_bert_label.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
9. Reiss BERT + Label Sentences - `reiss_main_bert_label.py`
    - Do Data: `python reiss_main_bert_label.py --do_data`
    - Training: `python reiss_main_bert_label.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 8`
    - Testing: `python reiss_main_bert_label.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
10. Reiss Action Char-Line Pairs BERT - `reiss_action_bert_base.py`
    - Do Data: `python reiss_action_bert_base.py --do_data`
    - Training: `python reiss_action_bert_base.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 8`
    - Testing: `python reiss_action_bert_base.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
11. Reiss Action BERT + LEAM - `reiss_action_main_leam_bert_label.py`
    - Do Data: `python reiss_action_main_leam_bert_label.py --do_data`
    - Training: `python reiss_action_main_leam_bert_label.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 8`
    - Testing: `python reiss_action_main_leam_bert_label.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`
12. Reiss Action BERT + Label Sentences - `reiss_action_main_bert_label.py`
    - Do Data: `python reiss_action_main_bert_label.py --do_data`
    - Training: `python reiss_action_main_bert_label.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 5 --train_batch_size 8`
    - Testing: `python reiss_action_main_bert_label.py --do_test --save_best --do_lower_case --monitor valid_loss --mode min`

## Major Software Requirements
The following is a list of packages and versions required to run this codebase. A full list of requirements can be found in the `requirements.txt` file at the root of this repository
- Python=3.6
- torch==1.2.0+cu92
- pandas==0.25.1
- pytorch-transformers==1.1.0
- transformers==2.0.0
- matplotlib==3.1.1
- scikit-learn==0.21.3
- scipy==1.3.1

### Code citations
This work wouldn't have been possible without the baseline codebase to start this project as well as the dependences used within the codebase from the following sources. Special thanks to all of the following repositories.
* The code for the research I am basing this project on, [Modeling Label Semantics For Predicting Emotional Reactions](https://arxiv.org/pdf/2006.05489.pdf) (as mentioned previously) can be found here: https://github.com/StonyBrookNLP/emotion-label-semantics
* The code for multi-label classification with a BERT pretrained model was originally taken from:
https://github.com/lonePatient/Bert-Multi-Label-Text-Classification
* The code for the baselines in [Modeling Naive Psychology of Characters in Simple Commonsense Stories](https://uwnlp.github.io/storycommonsense/), was taken from:
https://github.com/atcbosselut/scs-baselines
* The following repository - https://github.com/guoyinwang/LEAM helped in understanding the LEAM architecture. The previous research I am basing this project on reimplemented this framework in Pytorch.
* The code for an additional baseline - [Ranking and Selecting Multi-Hop Knowledge Paths to Better PredictHuman Needs](https://www.aclweb.org/anthology/N19-1368.pdf) was referenced from - https://github.com/debjitpaul/Multi-Hop-Knowledge-Paths-Human-Needs
