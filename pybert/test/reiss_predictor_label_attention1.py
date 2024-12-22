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
                 epoch_metrics,
                 batch_metrics
                 ):
        self.model = model
        self.logger = logger
        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model)



    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.result = {}
        print (self.epoch_metrics)
        for metric in self.epoch_metrics:
            print (metric)
            metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()


    def predict(self,data):
        pbar = ProgressBar(n_total=len(data))
        self.epoch_reset()
        self.model.eval()


        all_predicted = np.zeros((1, 19)) #zero array of one row and 6 columns
        all_true = np.zeros((1, 19)) #zero array of one row and 6 columns

        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids_sent_char,input_mask_sent_char, segment_ids_sent_char, input_ids_label, input_mask_label,segment_ids_label,label_ids = batch
                
                logits = self.model(input_ids_sent_char,input_mask_sent_char, segment_ids_sent_char, input_ids_label, input_mask_label,segment_ids_label)

                logits_detach = logits.cpu().detach()
                
                m = torch.nn.Sigmoid()
                #run the sigmoids function over the logits
                prob = m(logits_detach.round())

                all_predicted = np.concatenate((all_predicted, prob.numpy()), axis=0)

                all_true = np.concatenate((all_true, label_ids.cpu().numpy()), axis=0)

                self.outputs.append(logits.cpu().detach())
                self.targets.append(label_ids.cpu().detach())
                pbar.batch_step(step=step,info = {},bar_type='Evaluating')

            self.outputs = torch.cat(self.outputs, dim = 0).cpu().detach()
            self.targets = torch.cat(self.targets, dim = 0).cpu().detach()
                        
            print("------------- Test result --------------")


            total_checked = 0
            total_correct = 0
            true_positive_by_label = np.zeros(all_predicted.shape[1])
            false_positive_by_label = np.zeros(all_predicted.shape[1])
            false_negative_by_label = np.zeros(all_predicted.shape[1])

            for i in range(all_predicted.shape[0]):
                current_logit = all_predicted[i]
                current_label = all_true[i]
                for j in range(current_logit.shape[0]):
                    if current_logit[j] >= 0.5:
                        current_logit[j] = 1
                    else:
                        current_logit[j] = 0
                for j in range(current_logit.shape[0]):
                    if current_logit[j] == 1:
                        total_checked += 1
                        if current_label[j] == 1:
                            total_correct += 1
                            true_positive_by_label[j] += 1
                        else:
                            false_positive_by_label[j] += 1
                    elif current_label[j] == 1:
                        false_negative_by_label[j] += 1

            print("Total checked: ", total_checked)
            print("Total correct: ", total_correct)
            print("Total accuracy: ", total_correct / total_checked)
            
            micro_precision = np.sum(true_positive_by_label) / (np.sum(true_positive_by_label) + np.sum(false_positive_by_label))
            micro_recall = np.sum(true_positive_by_label) / (np.sum(true_positive_by_label) + np.sum(false_negative_by_label))
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

            print("Micro precision: ", micro_precision)
            print("Micro recall: ", micro_recall)
            print("Micro F1: ", micro_f1)



            # self.epoch_metrics = self.epoch_metrics[1:]
            # if self.epoch_metrics:
            #     for metric in self.epoch_metrics:
            #         metric(logits=self.outputs, target=self.targets)
            #         value = metric.value()

            #         if value:
            #             self.result[f'test_{metric.name()}'] = value
            
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

            return micro_f1, all_predicted, all_true