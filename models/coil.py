import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet,CosineIncrementalNet,SimpleCosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
import ot
from utils.data_manager import DataManager, x_u_split, SSL100, TransformFixMatch,transforms, AverageMeter,interleave,de_interleave, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from convs.ema import ModelEMA
import time
import logging
from torch import nn
import copy
EPSILON = 1e-8


batch_size = 64
memory_size = 2000
T = 2
num_workers=8
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

class COIL(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleCosineIncrementalNet(args['convnet_type'], False)
        self.data_manager=None
        self.nextperiod_initialization=None
        self.sinkhorn_reg=args['sinkhorn']
        self.calibration_term=args['calibration_term']
        self.args=args
        self.ema_model = ModelEMA(self._network, 0.999)
    def after_task(self):

        if self._cur_task<9:
            self.nextperiod_initialization=self.solving_ot()
            self._old_network = self._network.copy().freeze()
            self._known_classes = self._total_classes

       
    def solving_ot(self):
        with torch.no_grad():
           
            each_time_class_num=self.data_manager.get_task_size(1)
            self._extract_class_means(self.data_manager,0,self._total_classes+each_time_class_num)
            former_class_means=torch.tensor(self._ot_prototype_means[:self._total_classes])
            next_period_class_means=torch.tensor(self._ot_prototype_means[self._total_classes:self._total_classes+each_time_class_num])
            Q_cost_matrix=torch.cdist(former_class_means,next_period_class_means,p=self.args['norm_term'])
            #solving ot
            _mu1_vec=torch.ones(len(former_class_means))/len(former_class_means)*1.0
            _mu2_vec=torch.ones(len(next_period_class_means))/len(former_class_means)*1.0
            T=ot.sinkhorn(_mu1_vec,_mu2_vec,Q_cost_matrix,self.sinkhorn_reg) 
            T=torch.tensor(T).float().cuda()
            transformed_hat_W=torch.mm(T.T,F.normalize(self._network.fc.weight, p=2, dim=1))
            oldnorm=(torch.norm(self._network.fc.weight,p=2,dim=1))
            newnorm=(torch.norm(transformed_hat_W*len(former_class_means),p=2,dim=1))
            meannew=torch.mean(newnorm)
            meanold=torch.mean(oldnorm)
            gamma=meanold/meannew
            self.calibration_term=gamma
            self._ot_new_branch=transformed_hat_W*len(former_class_means)*self.calibration_term
        return transformed_hat_W*len(former_class_means)*self.calibration_term


    def solving_ot_to_old(self):
        current_class_num=self.data_manager.get_task_size(self._cur_task)
        self._extract_class_means_with_memory(self.data_manager,self._known_classes,self._total_classes)
        former_class_means=torch.tensor(self._ot_prototype_means[:self._known_classes])
        next_period_class_means=torch.tensor(self._ot_prototype_means[self._known_classes:self._total_classes])
        Q_cost_matrix=torch.cdist(next_period_class_means,former_class_means,p=self.args['norm_term'])+EPSILON #in case of numerical err
        _mu1_vec=torch.ones(len(former_class_means))/len(former_class_means)*1.
        _mu2_vec=torch.ones(len(next_period_class_means))/len(former_class_means)*1.
        T=ot.sinkhorn(_mu2_vec,_mu1_vec,Q_cost_matrix,self.sinkhorn_reg) 
        T=torch.tensor(T).float().cuda()
        transformed_hat_W=torch.mm(T.T,F.normalize(self._network.fc.weight[-current_class_num:,:], p=2, dim=1))
        return transformed_hat_W*len(former_class_means)*self.calibration_term


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        self._network.update_fc(self._total_classes, self.nextperiod_initialization)
        self.data_manager=data_manager

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        self.lamda = self._known_classes / self._total_classes
        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train',appendent=self._get_memory())
        train_labeled_idxs, train_unlabeled_idxs = x_u_split(self._known_classes, self._total_classes,
                                                             train_dataset.labels)
        transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
        ])
        train_labeled_dataset = SSL100('./data', train_dataset, train_labeled_idxs, train=True,
                                       transform=transform_labeled)

        train_unlabeled_dataset = SSL100('./data', train_dataset, train_unlabeled_idxs, train=True,
                                         transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

        self.labeled_trainloader = DataLoader(train_labeled_dataset, sampler=RandomSampler(train_labeled_dataset),
                                              batch_size=batch_size,
                                              num_workers=num_workers, drop_last=True)
        self.unlabeled_trainloader = DataLoader(train_unlabeled_dataset, sampler=RandomSampler(train_unlabeled_dataset),
                                                batch_size=batch_size * 7,
                                                num_workers=num_workers, drop_last=True)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers)
        self._train(self.labeled_trainloader, self.unlabeled_trainloader, self.test_loader, self.ema_model)

        self._reduce_exemplar(data_manager, memory_size//self._total_classes)
        self._construct_exemplar(data_manager, memory_size//self._total_classes)

    def _train(self,labeled_trainloader, unlabeled_trainloader, test_loader,ema_model):
        self._network.to(self._device)
        no_decay = ['bias', 'bn']
        if self._old_network is not None:
            self._old_network.to(self._device)
        grouped_parameters = [
            {'params': [p for n, p in self._network.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0005},
            {'params': [p for n, p in self._network.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optim.SGD(grouped_parameters, lr=0.03,
                              momentum=0.9, nesterov=True)

        if self._cur_task == 0:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 0, 160 * 80)
            self._init_train(labeled_trainloader, unlabeled_trainloader, test_loader, optimizer, scheduler, ema_model)
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 0, 160 * 110)
            self._update_representation(labeled_trainloader, unlabeled_trainloader,
                                        test_loader, optimizer, scheduler,
                                        ema_model)

    def _init_train(self, labeled_trainloader, unlabeled_trainloader, test_loader, optimizer, scheduler,
                               ema_model):

        labeled_iter = iter(labeled_trainloader)
        unlabeled_iter = iter(unlabeled_trainloader)

        global best_acc
        end = time.time()

        self._network.train()

        for epoch in range(160):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            losses_x = AverageMeter()
            losses_u = AverageMeter()
            mask_probs = AverageMeter()

            p_bar = tqdm(range(80))

            weight_ot_init = max(1. - (epoch / 2) ** 2, 0)
            weight_ot_co_tuning = (epoch / 80) ** 2.

            for batch_idx in range(80):
                try:
                    inputs_x, targets_x = labeled_iter.next()
                except:
                    labeled_iter = iter(labeled_trainloader)
                    inputs_x, targets_x = labeled_iter.next()
                try:
                    (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                except:

                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                data_time.update(time.time() - end)
                batch_size = inputs_x.shape[0]

                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * 7 + 1).to(self._device)

                targets_x = targets_x.to(self._device)
                output = self._network(inputs)
                logits = output['logits']
                onehots = target2onehot(targets_x, self._total_classes)

                logits = de_interleave(logits, 2 * 7 + 1)

                logits_x = logits[:batch_size]

                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits
                Lx_clf = F.cross_entropy(logits_x, targets_x)

                pseudo_label = torch.softmax(logits_u_w[:, self._known_classes:].detach() / 1, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)

                mask = max_probs.ge(0.95).float()

                Lu_clf = (F.cross_entropy(logits_u_s[:, self._known_classes:], targets_u,
                                          reduction='none') * mask).mean()
                clf_loss = Lx_clf + Lu_clf

                if self._old_network is not None:

                    # old_logits = self._old_network(inputs)['logits'].detach()
                    logits_old = self._old_network(inputs)['logits'].detach()
                    logits_old = de_interleave(logits_old, 2 * 7 + 1)

                    logits_old_x = logits_old[:batch_size]

                    logits_old_u_w, logits_old_u_s = logits_old[batch_size:].chunk(2)

                    hat_pai_k = F.softmax(logits_old_x / T, dim=1)
                    log_pai_k = F.log_softmax(logits_x[:, :self._known_classes] / T, dim=1)
                    hat_pai_k1 = F.softmax(logits_old_u_w / T, dim=1)
                    log_pai_k1 = F.log_softmax(logits_u_w[:, :self._known_classes] / T, dim=1)
                    hat_pai_k2 = F.softmax(logits_old_u_s / T, dim=1)
                    log_pai_k2 = F.log_softmax(logits_u_s[:, :self._known_classes] / T, dim=1)

                    distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1)) - torch.mean(
                        torch.sum(hat_pai_k1 * log_pai_k1, dim=1)) - torch.mean(
                        torch.sum(hat_pai_k2 * log_pai_k2, dim=1))

                    if epoch < 1:
                        output = output['features']
                        output = de_interleave(output, 2 * 7 + 1)
                        output_x = output[:batch_size]

                        features_x = F.normalize(output_x, p=2, dim=1)

                        current_logit_new = F.log_softmax(logits_x[:, self._known_classes:] / T, dim=1)

                        new_logit_by_wnew_init_by_ot = F.linear(features_x,
                                                                F.normalize(self._ot_new_branch, p=2, dim=1))
                        new_logit_by_wnew_init_by_ot = F.softmax(new_logit_by_wnew_init_by_ot / T, dim=1)

                        new_branch_distill_loss = -torch.mean(
                            torch.sum(current_logit_new * new_logit_by_wnew_init_by_ot, dim=1))
                        loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda) + \
                               0.001 * (weight_ot_init * new_branch_distill_loss)
                    else:
                        output = output['features']
                        output = de_interleave(output, 2 * 7 + 1)
                        output_x = output[:batch_size]

                        features_x = F.normalize(output_x, p=2, dim=1)

                        if batch_idx % 30 == 0:
                            with torch.no_grad():
                                self._ot_old_branch = self.solving_ot_to_old()
                        old_logit_by_wold_init_by_ot = F.linear(features_x,
                                                                F.normalize(self._ot_old_branch, p=2, dim=1))
                        old_logit_by_wold_init_by_ot = F.log_softmax(old_logit_by_wold_init_by_ot / T, dim=1)

                        old_branch_distill_loss = -torch.mean(
                            torch.sum(hat_pai_k * old_logit_by_wold_init_by_ot, dim=1))
                        loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda) + \
                               self.args['reg_term'] * (weight_ot_co_tuning * old_branch_distill_loss)
                else:
                    loss = clf_loss
                loss.backward()
                losses.update(loss.item())
                losses_x.update(Lx_clf.item())
                losses_u.update(Lu_clf.item())
                optimizer.step()
                scheduler.step()
                self.ema_model.update(self._network)
                self._network.zero_grad()
                batch_time.update(time.time() - end)
                end = time.time()
                mask_probs.update(mask.mean().item())
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=200,
                        batch=batch_idx + 1,
                        iter=80,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg))
                p_bar.update()
            p_bar.close()


    def _update_representation(self, labeled_trainloader, unlabeled_trainloader, test_loader, optimizer, scheduler,ema_model):

        labeled_iter = iter(labeled_trainloader)
        unlabeled_iter = iter(unlabeled_trainloader)

        global best_acc
        end = time.time()

        self._network.train()

        for epoch in range(160):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            losses_x = AverageMeter()
            losses_u = AverageMeter()
            mask_probs = AverageMeter()

            p_bar = tqdm(range(80))

            weight_ot_init=max(1.-(epoch/2)**2,0)
            weight_ot_co_tuning=(epoch/80)**2.

            for batch_idx in range(80):
                try:
                    inputs_x, targets_x = labeled_iter.next()
                except:
                    labeled_iter = iter(labeled_trainloader)
                    inputs_x, targets_x = labeled_iter.next()
                try:
                    (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                except:

                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                data_time.update(time.time() - end)
                batch_size = inputs_x.shape[0]

                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * 7 + 1).to(self._device)

                targets_x = targets_x.to(self._device)
                output=self._network(inputs)
                logits = output['logits']
                onehots = target2onehot(targets_x, self._total_classes)

                logits = de_interleave(logits, 2 * 7 + 1)

                logits_x = logits[:batch_size]

                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits
                Lx_clf = F.cross_entropy(logits_x, targets_x)

                pseudo_label = torch.softmax(logits_u_w[:, self._known_classes:].detach() / 1, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)

                mask = max_probs.ge(0.95).float()

                Lu_clf = (F.cross_entropy(logits_u_s[:, self._known_classes:], targets_u,
                                          reduction='none') * mask).mean()
                clf_loss =  Lx_clf + Lu_clf

                if self._old_network is not None:
                    
                    # old_logits = self._old_network(inputs)['logits'].detach()
                    logits_old = self._old_network(inputs)['logits'].detach()
                    logits_old = de_interleave(logits_old, 2 * 7 + 1)

                    logits_old_x = logits_old[:batch_size]

                    logits_old_u_w, logits_old_u_s = logits_old[batch_size:].chunk(2)




                    hat_pai_k = F.softmax(logits_old_x / T, dim=1)
                    log_pai_k = F.log_softmax(logits_x[:, :self._known_classes] / T, dim=1)
                    hat_pai_k1 = F.softmax(logits_old_u_w / T, dim=1)
                    log_pai_k1 = F.log_softmax(logits_u_w[:, :self._known_classes] / T, dim=1)
                    hat_pai_k2 = F.softmax(logits_old_u_s / T, dim=1)
                    log_pai_k2 = F.log_softmax(logits_u_s[:, :self._known_classes] / T, dim=1)

                    distill_loss = -torch.mean(torch.sum(hat_pai_k * log_pai_k, dim=1))-torch.mean(torch.sum(hat_pai_k1 * log_pai_k1, dim=1))-torch.mean(torch.sum(hat_pai_k2 * log_pai_k2, dim=1))
                    
                    if epoch<1:
                        output = output['features']
                        output = de_interleave(output, 2 * 7 + 1)
                        output_x = output[:batch_size]

                        features_x = F.normalize(output_x, p=2, dim=1)

                        current_logit_new = F.log_softmax(logits_x[:, self._known_classes:] / T, dim=1)

                        new_logit_by_wnew_init_by_ot = F.linear(features_x, F.normalize(self._ot_new_branch, p=2, dim=1))
                        new_logit_by_wnew_init_by_ot = F.softmax(new_logit_by_wnew_init_by_ot / T, dim=1)

                        new_branch_distill_loss = -torch.mean(torch.sum(current_logit_new * new_logit_by_wnew_init_by_ot, dim=1))
                        loss=distill_loss * self.lamda + clf_loss * (1 - self.lamda) +\
                        0.001*(weight_ot_init*new_branch_distill_loss)
                    else:
                        output = output['features']
                        output = de_interleave(output, 2 * 7 + 1)
                        output_x = output[:batch_size]

                        features_x = F.normalize(output_x, p=2, dim=1)

                        if batch_idx%30==0:
                            with torch.no_grad():
                                self._ot_old_branch=self.solving_ot_to_old()
                        old_logit_by_wold_init_by_ot = F.linear(features_x, F.normalize(self._ot_old_branch, p=2, dim=1))
                        old_logit_by_wold_init_by_ot = F.log_softmax(old_logit_by_wold_init_by_ot / T, dim=1)


                        old_branch_distill_loss = -torch.mean(torch.sum(hat_pai_k* old_logit_by_wold_init_by_ot, dim=1))
                        loss=distill_loss * self.lamda + clf_loss * (1 - self.lamda) +\
                        self.args['reg_term']*(weight_ot_co_tuning*old_branch_distill_loss)
                else:
                    loss = clf_loss
                loss.backward()
                losses.update(loss.item())
                losses_x.update(Lx_clf.item())
                losses_u.update(Lu_clf.item())
                optimizer.step()
                scheduler.step()
                self.ema_model.update(self._network)
                self._network.zero_grad()
                batch_time.update(time.time() - end)
                end = time.time()
                mask_probs.update(mask.mean().item())
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=200,
                        batch=batch_idx + 1,
                        iter=80,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg))
                p_bar.update()
            p_bar.close()

    def _extract_class_means(self,data_manager,low,high):
        self._ot_prototype_means = np.zeros((data_manager.get_total_classnum(), self._network.feature_dim))
        with torch.no_grad():
            for class_idx in range(low, high):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                class_mean = class_mean / (np.linalg.norm(class_mean))
                self._ot_prototype_means[class_idx, :] = class_mean
        self._network.train()
    def _extract_class_means_with_memory(self,data_manager,low,high):
        
        self._ot_prototype_means = np.zeros((data_manager.get_total_classnum(), self._network.feature_dim))
        memoryx,memoryy=self._data_memory,self._targets_memory
        with torch.no_grad():
            for class_idx in range(0,low):
                idxes = np.where(np.logical_and(memoryy >= class_idx, memoryy < class_idx+1))[0]
                data, targets = memoryx[idxes],memoryy[idxes]
                #idx_dataset=TensorDataset(data,targets)
                #idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                _,_,idx_dataset=data_manager.get_dataset([], source='train', appendent=(data,targets) ,mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                class_mean = class_mean / np.linalg.norm(class_mean)
                self._ot_prototype_means[class_idx, :] = class_mean

            for class_idx in range(low, high):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                class_mean = class_mean / np.linalg.norm(class_mean)
                self._ot_prototype_means[class_idx, :] = class_mean
        self._network.train()
