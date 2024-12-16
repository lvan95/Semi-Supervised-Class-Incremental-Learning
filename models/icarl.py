import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.data_manager import DataManager,x_u_split, SSL100, TransformFixMatch,transforms, AverageMeter,interleave,de_interleave, get_cosine_schedule_with_warmup,_KD_loss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from convs.ema import ModelEMA
import time
import logging
EPSILON = 1e-8

init_epoch=200
init_lr=0.1
init_milestones=[60,120,170]
init_lr_decay=0.1
init_weight_decay=0.0005


epochs = 170
lrate = 0.1
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 64
weight_decay=2e-4
num_workers=8
T=2

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)




class iCaRL(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args['convnet_type'], False)
        self.ema_model = ModelEMA(self._network, 0.999)
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        if self._cur_task == 0:

            train_labeled_idxs, train_unlabeled_idxs = x_u_split(self._known_classes, self._total_classes,
                                                                 train_dataset.labels)


        else:
            b = self.exemplar_size
            train_labeled_idxs, train_unlabeled_idxs = x_u_split(self._known_classes, self._total_classes,
                                                                 train_dataset.labels[:5000])
            train_labeled_idxs = np.concatenate((train_labeled_idxs, np.arange(5000, b + 5000)))

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
                                                batch_size=batch_size ,
                                                num_workers=num_workers, drop_last=True)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers)
        self._train(self.labeled_trainloader, self.unlabeled_trainloader, self.test_loader, self.ema_model)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
    def _train(self, labeled_trainloader, unlabeled_trainloader, test_loader, ema_model):
        self._network.to(self._device)
        no_decay = ['bias', 'bn']
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
                optimizer, 0, 200 * 80)
            self._init_train(labeled_trainloader, unlabeled_trainloader, test_loader, optimizer, scheduler, ema_model)
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 0, 170 * 120)
            self._update_representation(labeled_trainloader, unlabeled_trainloader, test_loader, optimizer, scheduler,
                                        ema_model)





    def _init_train(self, labeled_trainloader, unlabeled_trainloader, test_loader, optimizer, scheduler, ema_model):
        labeled_iter = iter(labeled_trainloader)
        unlabeled_iter = iter(unlabeled_trainloader)

        global best_acc
        end = time.time()

        self._network.train()

        for epoch in range(200):

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            losses_x = AverageMeter()
            losses_u = AverageMeter()
            mask_probs = AverageMeter()
            p_bar = tqdm(range(80))
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
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * 1 + 1).to(self._device)

                targets_x = targets_x.to(self._device)

                logits = self._network(inputs)['logits']

                logits = de_interleave(logits, 2 *1 + 1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                pseudo_label = torch.softmax(logits_u_w.detach() / 1, dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(0.95).float()
                Lu = (F.cross_entropy(logits_u_s, targets_u,
                                      reduction='none') * mask).mean()
                loss = Lx + Lu
                loss.backward()
                losses.update(loss.item())
                losses_x.update(Lx.item())
                losses_u.update(Lu.item())
                optimizer.step()
                scheduler.step()
                ema_model.update(self._network)
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

    def _update_representation(self, labeled_trainloader, unlabeled_trainloader, test_loader, optimizer, scheduler,
                               ema_model):
        labeled_iter = iter(labeled_trainloader)
        unlabeled_iter = iter(unlabeled_trainloader)

        global best_acc
        end = time.time()

        self._network.train()

        for epoch in range(170):

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            losses_x = AverageMeter()
            losses_u = AverageMeter()
            mask_probs = AverageMeter()
            p_bar = tqdm(range(120))
            for batch_idx in range(120):
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
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * 1 + 1).to(self._device)

                targets_x = targets_x.to(self._device)

                logits = self._network(inputs)['logits']

                logits = de_interleave(logits, 2 * 1 + 1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                pseudo_label = torch.softmax(logits_u_w.detach() / 1, dim=-1)

                max_probs, targets_u = torch.max(pseudo_label, dim=-1)

                mask = max_probs.ge(0.95).float()
                Lu = (F.cross_entropy(logits_u_s, targets_u,
                                      reduction='none') * mask).mean()

                logits_old = self._old_network(inputs)['logits']
                logits_old = de_interleave(logits_old, 2 * 7 + 1)

                logits_old_x = logits_old[:batch_size]

                logits_old_u_w, logits_old_u_s = logits_old[batch_size:].chunk(2)
                del logits_old
                #Loss_kd = _KD_loss(logits_x[:, :self._known_classes], logits_old_x, T)
                Loss_kd1 = _KD_loss(logits_old_u_w, logits_u_w[:, :self._known_classes], T)
                Loss_kd2 = _KD_loss(logits_old_u_s, logits_u_s[:, :self._known_classes], T)
                loss = Lx + Lu + Loss_kd1 + Loss_kd2


                loss.backward()
                losses.update(loss.item())
                losses_x.update(Lx.item())
                losses_u.update(Lu.item())
                optimizer.step()
                scheduler.step()
                ema_model.update(self._network)
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

#     def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
#         prog_bar = tqdm(range(epochs))
#         for _, epoch in enumerate(prog_bar):
#             self._network.train()
#             losses = 0.
#             correct, total = 0, 0
#             for i, (_, inputs, targets) in enumerate(train_loader):
#                 inputs, targets = inputs.to(self._device), targets.to(self._device)
#                 logits = self._network(inputs)['logits']
#
#                 loss_clf=F.cross_entropy(logits,targets)
#                 loss_kd=_KD_loss(logits[:,:self._known_classes],self._old_network(inputs)["logits"],T)
#
#                 loss=loss_clf+loss_kd
#
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 losses += loss.item()
#
#                 _, preds = torch.max(logits, dim=1)
#                 correct += preds.eq(targets.expand_as(preds)).cpu().sum()
#                 total += len(targets)
#
#             scheduler.step()
#             train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
#             if epoch%5==0:
#                 test_acc = self._compute_accuracy(self._network, test_loader)
#                 info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
#                 self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc, test_acc)
#             else:
#                 info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
#                 self._cur_task, epoch+1, epochs, losses/len(train_loader), train_acc)
#             prog_bar.set_description(info)
#         logging.info(info)
# def _KD_loss(pred, soft, T):
#     pred = torch.log_softmax(pred/T, dim=1)
#     soft = torch.softmax(soft/T, dim=1)
#     return -1 * torch.mul(soft, pred).sum()/pred.shape[0]
