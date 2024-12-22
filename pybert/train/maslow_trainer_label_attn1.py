
import torch
from ..callback.progressbar import ProgressBar
from ..common.tools import restore_checkpoint,model_device
from ..common.tools import summary
from ..common.tools import seed_everything
from ..common.tools import AverageMeter
from torch.nn.utils import clip_grad_norm_
import numpy as np

class Trainer(object):
    def __init__(self,n_gpu,
                 model,
                 epochs,
                 logger,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 early_stopping,
                 epoch_metrics,
                 batch_metrics,
                 gradient_accumulation_steps,
                 grad_clip = 0.0,
                 verbose = 1,
                 fp16 = None,
                 resume_path = None,
                 training_monitor = None,
                 model_checkpoint = None
                 ):
        self.start_epoch = 1
        self.global_step = 0
        self.n_gpu = n_gpu
        self.model = model
        self.epochs = epochs
        self.logger =logger
        self.fp16 = fp16
        self.grad_clip = grad_clip
        self.verbose = verbose
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping

        self.epoch_metrics = epoch_metrics
        self.batch_metrics = batch_metrics
        self.model_checkpoint = model_checkpoint
        self.training_monitor = training_monitor
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.model, self.device = model_device(n_gpu = self.n_gpu, model=self.model)
        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        if resume_path:
            
            print ("Path resumed:")
            self.logger.info(f"\nLoading checkpoint: {resume_path}")
            
            resume_dict = torch.load(resume_path / 'training_info.bin')
            ##get the best checkpoint
            best = resume_dict['epoch']

            self.start_epoch = resume_dict['epoch']

            ## Get the model checkpoint with the best performance
            if self.model_checkpoint:
                self.model_checkpoint.best = best

            self.logger.info(f"\nCheckpoint '{resume_path}' and epoch {self.start_epoch} loaded")


    def epoch_reset(self):

        ##Emoty out the predicted labels, and target labels so that a new set of labels can be filled for a new epoch
        self.outputs = []
        self.targets = []
        self.result = {}
        
        for metric in self.epoch_metrics:
            metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()

    def save_info(self,epoch,best, predicted, true, train_true_labels):
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model":model_save,
                 'epoch':epoch,
                 'best':best,
                 'val_predicted' : predicted,
                 'val_true' : true,
                 'train_true' : train_true_labels}

        return state

    def valid_epoch(self,data):
        pbar = ProgressBar(n_total=len(data))
        self.epoch_reset()
        self.model.eval()


        all_predicted = np.zeros((1, 5)) #zero array of one row and 5 columns
        all_true = np.zeros((1, 5)) #zero array of one row and 5 columns

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
            
            true_positive_by_label = np.zeros(len(self.outputs[0]))
            false_positive_by_label = np.zeros(len(self.outputs[0]))
            false_negative_by_label = np.zeros(len(self.outputs[0]))
            for i in range(len(self.outputs)):
                for j in range(len(self.outputs[i])):
                    if self.outputs[i][j] > 0.5 and self.targets[i][j] == 1:
                        true_positive_by_label[j] += 1
                    elif self.outputs[i][j] > 0.5 and self.targets[i][j] == 0:
                        false_positive_by_label[j] += 1
                    elif self.outputs[i][j] <= 0.5 and self.targets[i][j] == 1:
                        false_negative_by_label[j] += 1
            micro_precision = np.sum(true_positive_by_label) / (np.sum(true_positive_by_label) + np.sum(false_positive_by_label))
            micro_recall = np.sum(true_positive_by_label) / (np.sum(true_positive_by_label) + np.sum(false_negative_by_label))
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            print("\nValid micro precision: ", micro_precision)
            print("Valid micro recall: ", micro_recall)
            print("Valid micro f1: ", micro_f1)

            loss = self.criterion(target = self.targets, output=self.outputs)
            
            self.result['valid_loss'] = loss.item()
            
            print("------------- valid result --------------")
            
            # self.epoch_metrics = self.epoch_metrics[1:]
            if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    metric(logits=self.outputs, target=self.targets)
                    value = metric.value()

                    if value:
                        self.result[f'valid_{metric.name()}'] = value
            
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

            return self.result, all_predicted, all_true

    def train_epoch(self,data):
        pbar = ProgressBar(n_total = len(data))
        tr_loss = AverageMeter()
        self.epoch_reset()

        all_predicted = np.zeros((1, 5)) #zero array of one row and 5 columns
        all_true = np.zeros((1, 5)) #zero array of one row and 5 columns

        for step,  batch in enumerate(data):
            self.batch_reset()
            self.model.train()

            batch = tuple(t.to(self.device) for t in batch)
            input_ids_sent_char,input_mask_sent_char, segment_ids_sent_char, input_ids_label, input_mask_label,segment_ids_label,label_ids = batch
            #get the output from the BERT pre-trained model

            logits = self.model(input_ids_sent_char,input_mask_sent_char, segment_ids_sent_char,input_ids_label, input_mask_label,segment_ids_label)

            loss = self.criterion(output=logits,target=label_ids)

            #detach the logits from the computation graph. It is not needed for backprop from here.
            logits_detach = logits.cpu().detach()
            
            m = torch.nn.Sigmoid()
            #run the sigmoids function over the logits
            prob = m(logits_detach.round())

            all_predicted = np.concatenate((all_predicted, prob.numpy()), axis=0)

            all_true = np.concatenate((all_true, label_ids.cpu().numpy()), axis=0)

            if len(self.n_gpu) >= 2:
                loss = loss.mean()

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            
            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_norm_(amp.master_params(self.optimizer), self.grad_clip)
            else:
                loss.backward()

                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.lr_scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            if self.batch_metrics:
                for metric in self.batch_metrics:
                    metric(logits = logits,target = label_ids)
                    self.info[metric.name()] = metric.value()
            
            self.info['loss'] = loss.item()
            
            tr_loss.update(loss.item(),n = 1)
            
            if self.verbose >= 1:
                pbar.batch_step(step= step,info = self.info, bar_type='Training')
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
            # break

        print("\n------------- train result --------------")
        # epoch metric
        self.outputs = torch.cat(self.outputs, dim =0).cpu().detach()
        self.targets = torch.cat(self.targets, dim =0).cpu().detach()

        true_positive_by_label = np.zeros(len(self.outputs[0]))
        false_positive_by_label = np.zeros(len(self.outputs[0]))
        false_negative_by_label = np.zeros(len(self.outputs[0]))
        for i in range(len(self.outputs)):
            for j in range(len(self.outputs[i])):
                if self.outputs[i][j] > 0.5 and self.targets[i][j] == 1:
                    true_positive_by_label[j] += 1
                elif self.outputs[i][j] > 0.5 and self.targets[i][j] == 0:
                    false_positive_by_label[j] += 1
                elif self.outputs[i][j] <= 0.5 and self.targets[i][j] == 1:
                    false_negative_by_label[j] += 1
        micro_precision = np.sum(true_positive_by_label) / (np.sum(true_positive_by_label) + np.sum(false_positive_by_label))
        micro_recall = np.sum(true_positive_by_label) / (np.sum(true_positive_by_label) + np.sum(false_negative_by_label))
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        print("\nTrain Micro Precision: ", micro_precision)
        print("Train Micro Recall: ", micro_recall)
        print("Train Micro F1: ", micro_f1)

        print ("Targets:")
        print (self.targets)
        ##Average loss for this epoch

        self.result['loss'] = tr_loss.avg

        print ("Epoch metrics:")
        # self.epoch_metrics = self.epoch_metrics[:]

        # print (self.epoch_metrics)

        # if self.epoch_metrics:

        #     for metric in self.epoch_metrics:

        #         print ("Metric is:")
        #         print (metric)
                

        #         # print ("                ")
        #         metric(logits=self.outputs, target=self.targets)
                
        if self.epoch_metrics:
            for metric in self.epoch_metrics:
                metric(logits=self.outputs, target=self.targets)
                value = metric.value()
                if value:
                    self.result[f'{metric.name()}'] = value
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return self.result, all_predicted, all_true


        # if "cuda" in str(self.device):
        #     torch.cuda.empty_cache()
        
        # return self.result, predicted_labels, true_labels

    def train(self,train_data,valid_data,seed):
        seed_everything(seed)
        print("model summary info: ")


# all_input_ids_sent, all_input_mask_sent, all_segment_ids_sent, all_input_ids_char, all_input_mask_char, all_segment_ids_char, all_input_ids_label, all_input_mask_label, all_segment_ids_label, all_label_ids

        for step, (input_ids_sent_char,input_mask_sent_char, segment_ids_sent_char, input_ids_label, input_mask_label,segment_ids_label,
        label_ids) in enumerate(train_data):

            input_ids_sent_char = input_ids_sent_char.to(self.device)
            input_mask_sent_char = input_mask_sent_char.to(self.device)
            segment_ids_sent_char = segment_ids_sent_char.to(self.device)
            input_ids_label = input_ids_label.to(self.device)
            input_mask_label = input_mask_label.to(self.device)
            segment_ids_label = segment_ids_label.to(self.device)

            summary(self.model,*(input_ids_sent_char,input_mask_sent_char, segment_ids_sent_char, input_ids_label, input_mask_label,segment_ids_label),show_input=True)
            break

        # ***************************************************************
        
        ##For each epoch, get the train and validation data performance
        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):

            self.logger.info(f"Epoch {epoch}/{self.epochs}")
            
            train_log, train_pred, train_true = self.train_epoch(train_data)

            valid_log, valid_pred, valid_true = self.valid_epoch(valid_data)

            logs = dict(train_log,**valid_log)

            print ("Logs are:")
            print (logs)

            show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key,value in logs.items()])
            
            self.logger.info(show_info)

            # save
            if self.training_monitor:
                self.training_monitor.epoch_step(logs) 

            # save model
            # Metric used for comparing the models is the validation loss
            if self.model_checkpoint:
                state = self.save_info(epoch,best=logs["valid_loss"], predicted=valid_pred, true=valid_true, train_true_labels=train_true)

                self.model_checkpoint.bert_epoch_step(current=logs[self.model_checkpoint.monitor],state = state)

            # early_stopping
            if self.early_stopping:
                self.early_stopping.epoch_step(epoch=epoch, current=logs[self.early_stopping.monitor])

                if self.early_stopping.stop_training:
                    break






