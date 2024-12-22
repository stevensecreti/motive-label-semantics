#encoding:utf-8
import torch
import numpy as np
from ..common.tools import model_device
from ..callback.progressbar import ProgressBar

class Predictor(object):
    def __init__(self,
                 model,
                 logger,
                 n_gpu,
                 batch_metrics,
                 epoch_metrics,
                 ):
        self.model = model
        self.logger = logger
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)
        self.batch_metrics = batch_metrics
        self.epoch_metrics = epoch_metrics


    def predict(self,data):
        pbar = ProgressBar(n_total=len(data))
        all_logits = None
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = logits.sigmoid()
                if all_logits is None:
                    all_logits = logits.detach().cpu().numpy()
                    all_label_ids = label_ids.detach().cpu().numpy()
                else:
                    all_logits = np.concatenate([all_logits,logits.detach().cpu().numpy()],axis = 0)
                    all_label_ids = np.concatenate([all_label_ids,label_ids.detach().cpu().numpy()],axis = 0)
                pbar.batch_step(step=step,info = {},bar_type='Testing')
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        

        total_checked = 0
        total_correct = 0
        true_positive_by_label = np.zeros(all_logits.shape[1])
        false_positive_by_label = np.zeros(all_logits.shape[1])
        false_negative_by_label = np.zeros(all_logits.shape[1])

        for i in range(all_logits.shape[0]):
            current_logit = all_logits[i]
            for j in range(current_logit.shape[0]):
                if current_logit[j] >= 0.5:
                    current_logit[j] = 1
                else:
                    current_logit[j] = 0
            true_label = all_label_ids[i]
            for j in range(current_logit.shape[0]):
                if current_logit[j] == true_label[j]:
                    total_correct += 1
                    if current_logit[j] == 1:
                        true_positive_by_label[j] += 1
                elif current_logit[j] == 1:
                    false_positive_by_label[j] += 1
                elif true_label[j] == 1:
                    false_negative_by_label[j] += 1
                total_checked += 1
        
        results = total_correct/total_checked
        print("Accuracy: " + str(round(results*100,2))+"%")

        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0

        #Change these for the motive prediction task
        label_names = ["physiological", "stability", "love", "esteem", "spiritual growth"]
        #Calculate micro-averaged precision, recall, and f1
        micro_precision = np.sum(true_positive_by_label)/(np.sum(true_positive_by_label)+np.sum(false_positive_by_label))
        micro_recall = np.sum(true_positive_by_label)/(np.sum(true_positive_by_label)+np.sum(false_negative_by_label))
        micro_f1 = 2*micro_precision*micro_recall/(micro_precision+micro_recall)
        print("Micro-averaged precision: " + str(round(micro_precision*100,2))+"%")
        print("Micro-averaged recall: " + str(round(micro_recall*100,2))+"%")
        print("Micro-averaged f1: " + str(round(micro_f1*100,2))+"%")

        #Calculate macro-averaged precision, recall, and f1
        for i in range(true_positive_by_label.shape[0]):
            precision = true_positive_by_label[i]/(true_positive_by_label[i]+false_positive_by_label[i])
            recall = true_positive_by_label[i]/(true_positive_by_label[i]+false_negative_by_label[i])
            f1 = 2*(precision*recall)/(precision+recall)
            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            #Uncomment line below to print precision, recall, and f1 for each label
            #print("Label " + label_names[i] + ": Precision: " + str(round(precision*100,2)) + "%, Recall: " + str(round(recall*100,2)) + "%, F1: " + str(round(f1*100,2)) + "%")

        avg_precision = avg_precision/true_positive_by_label.shape[0]
        avg_recall = avg_recall/true_positive_by_label.shape[0]
        avg_f1 = avg_f1/true_positive_by_label.shape[0]
        print("Macro-Average Precision: " + str(round(avg_precision*100,2)) + "%, Macro-Average Recall: " + str(round(avg_recall*100,2)) + "%, Macro-Average F1: " + str(round(avg_f1*100,2)) + "%")

        return micro_f1, all_logits, all_label_ids






