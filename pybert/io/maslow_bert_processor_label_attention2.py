import csv
import torch
import numpy as np
from ..common.tools import load_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar
from torch.utils.data import TensorDataset
from pytorch_transformers import BertTokenizer

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, text_c = None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid   = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label  = label

class InputFeature(object):
    '''
    A single set of features of data.
    '''
    def __init__(self,input_ids_sent_char,input_mask_sent_char, segment_ids_sent_char,input_len_sent_char,input_ids_label, input_mask_label,segment_ids_label,
        input_len_label, label_id):
        
        self.input_ids_sent_char = input_ids_sent_char
        self.input_mask_sent_char  = input_mask_sent_char
        self.segment_ids_sent_char = segment_ids_sent_char
        self.input_len_sent_char = input_len_sent_char

        self.input_ids_label = input_ids_label
        self.input_mask_label  = input_mask_label
        self.segment_ids_label = segment_ids_label
        self.input_len_label = input_len_label
        self.label_id = label_id


class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self,vocab_path,do_lower_case):
        self.tokenizer = BertTokenizer(vocab_path,do_lower_case)

    def get_train(self, data_file):
        """Gets a collection of `InputExample`s for the train set."""
        return self.read_data(data_file)

    def get_dev(self, data_file):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.read_data(data_file)

    def get_test(self,lines):
        return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["physiological", "stability", "love", "esteem", "spiritual growth"]

    @classmethod
    def read_data(cls, input_file,quotechar = None):
        """Reads a tab separated value file."""
        if 'pkl' in str(input_file):
            lines = load_pickle(input_file)
        else:
            lines = input_file
        return lines

    def truncate_seq_pair(self,tokens_a,tokens_b,max_length):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def create_examples(self,lines,example_type,cached_examples_file):
        '''
        Creates examples for data
        '''
        maslow = ["physiological", "stability", "love", "esteem", "spiritual growth"]

        pbar = ProgressBar(n_total = len(lines))

        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
            
        else:
            examples = []

            for i,line in enumerate(lines):
                guid = '%s-%d'%(example_type,i)
                print (line[0].split("//"))
                #text_a contains both the sentence and context separated by the [SEP] token
                text_a = line[0].split("//")[0] + " [SEP] " + line[0].split("//")[1] 
                
                label = line[1]
                char = line[0].split("//")[2]

                print ("Text a:")
                print (text_a)
                # print ("Text b:")
                # print (text_b)
                print ("Guid:")
                print (guid)
                # print ("Label:")
                # print (label)
                print ("Character:")
                print (char)

                char_labels = " ".join([char + " needs " + lab + ". [SEP]"  for lab in maslow]).strip()
                ##append char_labels to text_b separated by [SEP]
                
                #True label for this sample
                if isinstance(label,str):
                    label = [np.float(x) for x in label.split(",")]
                else:
                    label = [np.float(x) for x in list(label)]
                
                print ("Text a:")
                print (text_a)
                # print ("Text b:")
                # print (text_b)
                print ("Labels:")
                print (label)
                print ("               ")

                # for sample_label in label_ids:
                unique, counts = np.unique(label, return_counts=True)
                
                for l in unique:
                    if (l != 0) and (l != 1):
                        print ("Found error")
                        print (label)
                        print ("          ")
                        # print (label_ids)
                        print ("----------")


                # text_b = None
                example = InputExample(guid = guid,text_a = text_a,text_b=char_labels, label=label)

                examples.append(example)

                pbar.batch_step(step=i,info={},bar_type='create examples')
                
                # break

            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        
        return examples

    def tokenize_labels(self, labels_all):

        labels_list = labels_all.split(". [SEP]")
        print (labels_list)

        labels_tokenized = ['[CLS]']

        for label in labels_list:
            labels_tokenized += self.tokenizer.tokenize(label) + ['[SEP]']

        #exclude the double SEP token
        return labels_tokenized[:-1]


    def create_features(self,examples,max_seq_len,cached_features_file):
        '''
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        '''
        pbar = ProgressBar(n_total=len(examples))
        if cached_features_file.exists():
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)

        else:
            features = []
            
            for ex_id,example in enumerate(examples):
                #text_a contains both the sentence and context
                tokens_a = self.tokenizer.tokenize(example.text_a)
                
                if len(tokens_a) > max_seq_len - 2:
                    tokens_a = tokens_a[:max_seq_len - 2]

                #tokens b contain the label and character sentences
                label_id = example.label

                # tokens label come from text_b
                tokens_label = self.tokenize_labels(example.text_b)

                print ("Tokenized labels are:")
                print (tokens_label)
                
                #sentence and char are separated by a [SEP] token and a [SEP] comes at the end too
                tokens_sent_char = ['[CLS]'] + tokens_a + ['[SEP]']

                segment_ids_sent_char = [0] * len(tokens_sent_char)

                segment_ids_label = [0] * len(tokens_label)
                
                print ("Tokens sent and char:")
                print (tokens_sent_char)
                print ("Segment ids sent:")
                print (segment_ids_sent_char)
                print ("                   ")

                print ("Tokens label:")
                print (tokens_label)
                print ("segment ids label:")
                print (segment_ids_label)
                print ("-----------------------")

                ##For sentence
                input_ids_sent_char = self.tokenizer.convert_tokens_to_ids(tokens_sent_char)

                input_mask_sent_char = [1] * len(input_ids_sent_char)
                padding_sent_char = [0] * (max_seq_len - len(input_ids_sent_char))
                input_len_sent_char = len(input_ids_sent_char)

                input_ids_sent_char   += padding_sent_char
                input_mask_sent_char  += padding_sent_char
                segment_ids_sent_char += padding_sent_char

                #For label
                input_ids_label = self.tokenizer.convert_tokens_to_ids(tokens_label)
                input_mask_label = [1] * len(input_ids_label)
                padding_label = [0] * (max_seq_len - len(input_ids_label))
                input_len_label = len(input_ids_label)

                input_ids_label   += padding_label
                input_mask_label  += padding_label
                segment_ids_label += padding_label

                # print (len(segment_ids))
                assert len(input_ids_sent_char) == max_seq_len
                assert len(input_mask_sent_char) == max_seq_len
                assert len(segment_ids_sent_char) == max_seq_len

                assert len(input_ids_label) == max_seq_len
                assert len(input_mask_label) == max_seq_len
                assert len(segment_ids_label) == max_seq_len

                feature = InputFeature(input_ids_sent_char = input_ids_sent_char,
                                       input_mask_sent_char = input_mask_sent_char,
                                       segment_ids_sent_char = segment_ids_sent_char,
                                       input_len_sent_char = input_len_sent_char,#
                                       input_ids_label = input_ids_label,
                                       input_mask_label = input_mask_label,
                                       segment_ids_label = segment_ids_label,
                                       input_len_label = input_len_label,
                                       label_id = label_id)
                features.append(feature)
                pbar.batch_step(step=ex_id, info={}, bar_type='create features')
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self,features,is_sorted = False):
        # Convert to Tensors and build dataset
        if is_sorted:
            logger.info("sorted data by th length of input")

            features = sorted(features,key=lambda x:x.input_len_sent_char,reverse=True)
        
        all_input_ids_sent_char = torch.tensor([f.input_ids_sent_char for f in features], dtype=torch.long)
        all_input_mask_sent_char = torch.tensor([f.input_mask_sent_char for f in features], dtype=torch.long)
        all_segment_ids_sent_char = torch.tensor([f.segment_ids_sent_char for f in features], dtype=torch.long)

        all_input_ids_label = torch.tensor([f.input_ids_label for f in features], dtype=torch.long)
        all_input_mask_label = torch.tensor([f.input_mask_label for f in features], dtype=torch.long)
        all_segment_ids_label = torch.tensor([f.segment_ids_label for f in features], dtype=torch.long)

        # all_label_ids_sent = torch.tensor([f.label_id for f in features],dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features],dtype=torch.long)
        dataset = TensorDataset(all_input_ids_sent_char, all_input_mask_sent_char, all_segment_ids_sent_char, all_input_ids_label, all_input_mask_label, all_segment_ids_label, all_label_ids)
        return dataset

