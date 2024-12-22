import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.utils.rnn as rnn_utils
import torch

def get_attention_score(x_ids, label_ids, mask):

        # print (x_ids.size())
        # print (label_ids.size())
        G = F.relu(torch.bmm(x_ids, label_ids))
        # print (G.size())

        # G = G.unsqueeze(1)

        # print (G.size())

        # Att_v = F.relu(conv_layer(G).squeeze(3))        

        # Att_v_tran = torch.transpose(Att_v, 1, 2)

        # print ("Size of Att_v:")
        # print (Att_v_tran.size())

        max_label =  F.max_pool1d(G, G.shape[2]).squeeze(2)

        # print (max_label.size())

        max_label = max_label.masked_fill(mask == 0, -1e10)

        attention_scores = F.softmax(max_label, dim=1).unsqueeze(1)
        # print (attention_scores)
        # print (attention_scores.size())

        return attention_scores

class BertForMultiLable(BertPreTrainedModel):
    def __init__(self, config):

        super(BertForMultiLable, self).__init__(config)
        
        # print (config)
        # config = 
        # config["output_hidden_states"] = True
        self.bert = BertModel(config)
        # self.RNN1 = nn.LSTM(input_size=768, hidden_size=384, num_layers=2, bidirectional=True, dropout = 0.5, batch_first=True)
        # self.RNN2 = nn.LSTM(input_size=768, hidden_size=384, num_layers=2, bidirectional=True, dropout = 0.5, batch_first=True)
        print ("Number of labels:")
        print (config.num_labels)
        print ("                         ")
        self.dropout = nn.Dropout(0.5)
        self.linear_sent_char = nn.Linear(2*384, config.num_labels)

        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.apply(self.init_weights)
        self.init_weights()


# all_input_ids_sent, all_input_mask_sent, all_segment_ids_sent, all_input_ids_char, all_input_mask_char, all_segment_ids_char, all_input_ids_label, all_input_mask_label, all_segment_ids_label, all_label_ids

    def forward(self, input_ids_sent_char,input_mask_sent_char, segment_ids_sent_char,input_ids_label, input_mask_label,segment_ids_label):
        # print ("Input ids:")
        # print (input_ids_sent_char.size())
        # print ("Segment ids:")
        # print (segment_ids_sent_char.size())
        # print ("Mask:")
        # print (input_mask_sent_char)
        # print (input_mask_sent_char.size())
        # print ("-------------------------------------------------")

        # print (len(self.bert(input_ids_sent_char, token_type_ids=segment_ids_sent_char, attention_mask=input_mask_sent_char, head_mask=None)))

        sent_char_final_layer, sent_char_pooled = self.bert(input_ids_sent_char, token_type_ids=segment_ids_sent_char, attention_mask=input_mask_sent_char, head_mask=None)

        # print (input_ids_char.size())

        # char_final_layer, char_pooled = self.bert(input_ids_char, token_type_ids=segment_ids_char, attention_mask=input_mask_char, head_mask=None)

        label_final_layer, label_pooled = self.bert(input_ids_label, token_type_ids=segment_ids_label, attention_mask=input_mask_label, head_mask=None)

        # label_enc_downsize = self.linear_label(label_enc_layers)
        x_label_ids_emb_trans = torch.transpose(label_final_layer, 1, 2)
        
        #save a copy of the sentence encoding to calculate the weighted sum later
        sent_char_enc_copy = sent_char_final_layer
        #save a copy of the context encoding to calculate the weighed sum later
        # ctx_enc_copy = char_final_layer

        #Normalize the label embeddings with l2 normalization
        x_label_ids_emb_trans = F.normalize(x_label_ids_emb_trans, dim=1, p=2)
        
        #Normalize the sentence embeddings with l2 normalization
        sent_char_enc_normalize = F.normalize(sent_char_final_layer, dim=2, p=2)
          
        # Normalize the context embeddings with l2 normalization
        # ctx_enc_normalize = F.normalize(char_final_layer, dim=2, p=2)     

        attn_score_sent_char = get_attention_score(sent_char_enc_normalize, x_label_ids_emb_trans, input_mask_sent_char)

        # attn_score_ctx = get_attention_score(ctx_enc_normalize,x_label_ids_emb_trans, input_mask_char)

        sent_char_attn = torch.bmm(attn_score_sent_char, sent_char_enc_copy).squeeze(1)

        linear_h_e = self.linear_sent_char(sent_char_attn)

        return linear_h_e

    def unfreeze(self,start_layer,end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())
        
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)