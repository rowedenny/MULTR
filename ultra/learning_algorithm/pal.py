from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils

"""
Model class of Position Bias Aware Learning Framework (PAL).

:Reference:
#. `Huifeng Guo et al, 2019. PAL: a position-bias aware learning framework for CTR prediction in live recommender
    systems <https://dl.acm.org/citation.cfm?id=3347033&dl=ACM&coll=DL>`_.

"""


class PositionModel(nn.Module):
    def __init__(self, input_size, max_num_position):
        super(PositionModel, self).__init__()

        self.input_size = input_size
        self.max_num_position = max_num_position

        self.position_embedding = nn.Embedding(max_num_position, input_size)
        self.pos_model = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )

    def forward(self, pos_indices):
        pos_emb = self.position_embedding(pos_indices)
        prob_seen = self.pos_model(pos_emb)
        return prob_seen


class PAL(BaseAlgorithm):
    def __init__(self, data_set, exp_settings):
        print('Build PAL model')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,                 # Learning rate.
            max_gradient_norm=5.0,            # Clip gradients to this norm.
            loss_func='softmax_cross_entropy',            # Select Loss function
            # Set strength for L2 regularization.
            l2_loss=0.0,
            grad_strategy='ada',            # Select gradient strategy
        )

        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.cuda = torch.device('cuda')
        self.train_summary = {}
        self.eval_summary = {}
        self.is_training = "is_train"
        print(exp_settings['learning_algorithm_hparams'])
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        if 'selection_bias_cutoff' in self.exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.feature_size = data_set.feature_size

        self.model = self.create_model(self.feature_size)
        self.pos_model = PositionModel(input_size=64, max_num_position=100)
        if self.is_cuda_avail:
            self.model = self.model.to(device=self.cuda)
            self.pos_model = self.pos_model.to(device=self.cuda)
        self.max_candidate_num = exp_settings['max_candidate_num']

        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0

        # Feeds for inputs.
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))

        self.optimizer_func = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, input_feed):
        self.global_step += 1
        self.model.train()
        self.pos_model.train()
        self.create_input_feed(input_feed, self.rank_list_size)

        # [batch_size, rank_list_size]
        train_output = self.ranking_model(self.model, self.rank_list_size)
        train_labels = self.labels
        self.loss = None

        pos_indices = torch.arange(self.rank_list_size).unsqueeze(0).to(device=self.cuda)
        prob_seen = self.pos_model(pos_indices).squeeze(-1)

        # [batch_size, rank_list_size]
        click_prob = prob_seen * train_output

        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss = self.sigmoid_loss_on_list(
                click_prob, train_labels)
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss = self.pairwise_loss_on_list(
                click_prob, train_labels)
        else:
            self.loss = self.softmax_loss(
                click_prob, train_labels)

        # params = tf.trainable_variables()
        params = self.model.parameters()
        if self.hparams.l2_loss > 0:
            loss_l2 = 0.0
            for p in params:
                loss_l2 += self.l2_loss(p)
            self.loss += self.hparams.l2_loss * loss_l2

        self.opt_step(self.optimizer_func, params)

        nn.utils.clip_grad_value_(train_labels, 1)
        print(" Loss %f at Global Step %d: " % (self.loss.item(), self.global_step))
        return self.loss.item(), None, self.train_summary

    def validation(self, input_feed, is_online_simulation=False):
        """Run a step of the model feeding the given inputs for validating process.
        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.
        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.
        """
        self.model.eval()
        self.create_input_feed(input_feed, self.max_candidate_num)
        with torch.no_grad():
            self.output = self.ranking_model(self.model,
                                             self.max_candidate_num)
        if not is_online_simulation:
            pad_removed_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, self.output)

            for metric in self.exp_settings['metrics']:
                topn = self.exp_settings['metrics_topn']
                metric_values = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(self.labels, pad_removed_output, None)
                for topn, metric_value in zip(topn,metric_values):
                    self.create_summary('%s_%d' % (metric, topn),
                                        '%s_%d' % (metric, topn), metric_value.item(), False)
        return None, self.output, self.eval_summary  # loss, outputs, summary.