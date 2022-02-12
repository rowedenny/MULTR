from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from ultra.learning_algorithm.base_algorithm import BaseAlgorithm
import ultra.utils

def sigmoid_prob(logits):
    return torch.sigmoid(logits - torch.mean(logits, -1, keepdim=True))


class DenoisingNet(nn.Module):
    def __init__(self, input_vec_size):
        super(DenoisingNet, self).__init__()
        self.linear_layer = nn.Linear(input_vec_size, 1)
        self.elu_layer = nn.ELU()
        self.propensity_net = nn.Sequential(self.linear_layer, self.elu_layer)
        self.list_size = input_vec_size

    def forward(self, input_list):
        output_propensity_list = []
        for i in range(self.list_size):
            # Add position information (one-hot vector)
            click_feature = [
                torch.unsqueeze(
                    torch.zeros_like(
                        input_list[i]), -1) for _ in range(self.list_size)]
            click_feature[i] = torch.unsqueeze(
                torch.ones_like(input_list[i]), -1)
            # Predict propensity with a simple network
            output_propensity_list.append(
                self.propensity_net(
                    torch.cat(
                        click_feature, 1)))

        return torch.cat(output_propensity_list, 1)


class UserSimulator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(UserSimulator, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_transform = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, hidden_size),
            nn.ELU(),
        )

        self.encoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                              num_layers=num_layers, bidirectional=True)
        self.decoder = nn.GRU(input_size=hidden_size * 4, hidden_size=hidden_size,
                              num_layers=num_layers, bidirectional=False)
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(100, hidden_size)
        self.click_embedding = nn.Embedding(2, self.hidden_size)

        self.proj_sess = nn.Linear(hidden_size * 2, hidden_size)
        self.proj_result = nn.Linear(hidden_size, hidden_size)
        self.click_linear = nn.Linear(hidden_size, 1)

    def predict_click_with_label(self, input_list, input_labels):
        """ Predict the logit of click prob for a batch of session with ground truth is known
        :param input_list L * [batch, D]  A list of  features for each query-doc pair
        :param input_labels: [batch, rank_list_size] the actual clicks in each session,
                             clicks comes from click stimulation
        :return:[Tensor]: size = [batch_size, rank_list_size, 2]
        """
        device = next(self.parameters()).device

        # [rank_list_size, batch_size, D]
        input_tensor = torch.stack(input_list, dim=0)
        input_tensor = input_tensor.float().to(device)

        rank_list_size, batch_size = input_tensor.size(0), input_tensor.size(1)
        input_tensor = self.input_transform(input_tensor)

        # query context encoder
        output, hidden = self.encoder(input_tensor)
        # [1, batch_size, D]
        sess_feature = output[-1].unsqueeze(0)
        sess_feature_proj = self.proj_sess(sess_feature).expand(rank_list_size, -1, -1)

        # result feature projection [rank_list_size, batch_size, D]
        result_feature_proj = self.proj_result(input_tensor)

        # position features
        position_indices = torch.arange(rank_list_size, dtype=torch.long, device=device)
        # [rank_list_size, D]
        position_features = self.position_embedding(position_indices)
        # [rank_list_size, batch_size, D]
        position_features = position_features.unsqueeze(1).expand(-1, batch_size, -1)

        # click features
        # padding = torch.full((batch_size, 1), self.padding_idx, dtype=torch.long).to(device)
        padding = torch.full((batch_size, 1), 0, dtype=torch.long).to(device)

        # [batch_size, rank_list_size]
        previous_clicks = torch.cat([padding, input_labels[:, :-1]], dim=1).long()
        # [rank_list_size, batch_size, D]
        click_features = self.click_embedding(previous_clicks).transpose(0, 1)

        # [rank_list_size, batch_size, D * 4]
        feature_input = torch.cat([sess_feature_proj, result_feature_proj, position_features, click_features], -1)

        # decode click
        decoder_hidden = Variable(torch.zeros(1 * self.num_layers, batch_size, self.hidden_size)).to(device)
        click_output, click_hidden = self.decoder(feature_input, decoder_hidden)
        click_output = self.dropout(click_output)
        # [rank_list_size, batch_size, 1]
        outputs_prob = F.sigmoid(self.click_linear(click_output))

        return outputs_prob

    def predict_click_without_label(self, input_list):
        device = next(self.parameters()).device

        # [rank_list_size, batch_size, D]
        input_tensor = torch.stack(input_list, dim=0)
        input_tensor = input_tensor.float().to(device)

        rank_list_size, batch_size = input_tensor.size(0), input_tensor.size(1)
        input_tensor = self.input_transform(input_tensor)

        # query context encoder
        output, hidden = self.encoder(input_tensor)
        # [1, batch_size, D]
        sess_feature = output[-1].unsqueeze(0)
        sess_feature_proj = self.proj_sess(sess_feature).expand(rank_list_size, -1, -1)

        # result feature projection [rank_list_size, batch_size, D]
        result_feature_proj = self.proj_result(input_tensor)

        # decode click
        outputs_prob, outputs_click = [], []
        decoder_hidden = None
        previous_clicks = Variable(torch.full([batch_size], 0, dtype=torch.long)).to(device)
        for i in range(rank_list_size):
            # [batch_size, D]
            position_features = self.position_embedding.weight[i]
            position_features = torch.unsqueeze(position_features, 0).expand(batch_size, -1)
            click_features = self.click_embedding(previous_clicks)

            input_features = torch.cat([sess_feature_proj[i], result_feature_proj[i],
                                        position_features, click_features], -1)
            # [1, batch_size, D * 4]
            input_features = torch.unsqueeze(input_features, 0)

            if decoder_hidden is None:
                decoder_hidden = Variable(torch.zeros(1 * self.num_layers, batch_size, self.hidden_size)).to(device)
            decoder_output, decoder_hidden = self.decoder(input_features, decoder_hidden)

            # [1, batch_size, D]
            decoder_output = self.dropout(decoder_output)
            # [batch_size, 1]
            click_now_prob = F.sigmoid(self.click_linear(decoder_output)).squeeze(0)

            click_now = self.sample(click_now_prob)
            previous_clicks = click_now

            outputs_prob.append(click_now_prob)
            outputs_click.append(click_now)

        # [rank_size, batch_size, 1]
        outputs_prob = torch.stack(outputs_prob, 0)
        # [rank_size, batch_size]
        outputs_click = torch.stack(outputs_click, 0)
        return outputs_prob, outputs_click

    def sample(self, click_prob):
        notclick_prob = 1.0 - click_prob
        weights = torch.cat([notclick_prob, click_prob], 1)
        clicks = torch.multinomial(weights, 1, replacement=True)
        return clicks.squeeze(1)

    def build(self, input_list, input_labels=None, return_probs=False):
        """
        :return: [List]: rank_list_size * [batch_size, *]
        """
        if input_labels is None:
            # generate click simulation
            # [rank_size, batch_size, 1], [rank_list, batch_size]
            outputs_prob, outputs_click = self.predict_click_without_label(input_list)

            if return_probs:
                return torch.unbind(outputs_prob, dim=0)
            else:
                # [rank_size, batch_size, 1]
                outputs_click = torch.unsqueeze(outputs_click, -1)
                return torch.unbind(outputs_click, dim=0)
        else:
            # train mode
            # [rank_list_size, batch_size, 1]
            outputs_prob = self.predict_click_with_label(input_list, input_labels)
            return torch.unbind(outputs_prob, dim=0)


class MULTR(BaseAlgorithm):
    def __init__(self, data_set, exp_settings, forward_only=False):
        print('Build Model-based Unbiased Learning to Rank Model')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.05,                   # Learning rate
            env_learning_rate=1e-3,
            max_gradient_norm=5.0,                # Clip gradients to this norm.
            ranker_loss_weight=1.0,               # Set the weight of unbiased ranking loss
            l2_loss=0.0,                          # Set strength for L2 regularization.
            env_l2_loss=1e-5,
            grad_strategy='ada',                  # Select gradient strategy for model
            logits_to_prob='softmax',             # the function used to convert logits to probability distributions
            max_propensity_weight=-1,             # Set maximum value for propensity weights
            loss_func='softmax_loss',             # Select Loss function
            propensity_learning_rate=-1.0,        # The learning rate for ranker (-1 means same with learning_rate).
            env_loss_func='softmax_cross_entropy_with_prob',           # Select Loss function
            sample_num=16,                        #
            sample_cutoff=10,
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
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.sample_num = self.hparams.sample_num
        self.sample_cutoff = self.hparams.sample_cutoff

        self.model = self.create_model(self.feature_size)
        self.user_simulator = UserSimulator(self.feature_size, 64, num_layers=1, dropout=0.4)
        self.propensity_model = DenoisingNet(self.rank_list_size)

        if self.is_cuda_avail:
            self.model = self.model.to(device=self.cuda)
            self.user_simulator = self.user_simulator.to(device=self.cuda)
            self.propensity_model = self.propensity_model.to(device=self.cuda)
        self.letor_features_name = "letor_features"
        self.letor_features = None
        self.docid_inputs_name = []  # a list of top documents
        self.labels_name = []  # the labels for the documents (e.g., clicks)
        self.docid_inputs = []  # a list of top documents
        self.labels = []  # the labels for the documents (e.g., clicks)
        for i in range(self.max_candidate_num):
            self.docid_inputs_name.append("docid_input{0}".format(i))
            self.labels_name.append("label{0}".format(i))

        if self.hparams.propensity_learning_rate < 0:
            self.propensity_learning_rate = float(self.hparams.learning_rate)
        else:
            self.propensity_learning_rate = float(self.hparams.propensity_learning_rate)
        self.learning_rate = float(self.hparams.learning_rate)
        self.global_step = 0

        # Select logits to prob function
        self.logits_to_prob = nn.Softmax(dim=-1)
        if self.hparams.logits_to_prob == 'sigmoid':
            self.logits_to_prob = sigmoid_prob

        self.optimizer_simulator = torch.optim.Adam(self.user_simulator.parameters(),
                                                    lr=self.hparams.env_learning_rate,
                                                    weight_decay=self.hparams.env_l2_loss)
        self.optimizer_func = torch.optim.Adagrad
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD

        print('Loss Function is ' + self.hparams.loss_func)
        # Select loss function
        self.loss_func = None
        if self.hparams.loss_func == 'sigmoid_loss':
            self.loss_func = self.sigmoid_loss_on_list
        elif self.hparams.loss_func == 'pairwise_loss':
            self.loss_func = self.pairwise_loss_on_list
        else:  # softmax loss without weighting
            self.loss_func = self.softmax_loss

        print('Environment Loss Function is ' + self.hparams.env_loss_func)
        self.env_loss_func = None
        if self.hparams.env_loss_func == 'softmax_cross_entropy_with_prob':
            self.env_loss_func = self.softmax_cross_entropy_with_prob
        else:
            raise NotImplementedError

    def separate_gradient_update(self):
        denoise_params = self.propensity_model.parameters()
        ranking_model_params = self.model.parameters()
        # Select optimizer

        if self.hparams.l2_loss > 0:
            # for p in denoise_params:
            #    self.exam_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
            for p in ranking_model_params:
                self.rank_loss += self.hparams.l2_loss * self.l2_loss(p)
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * (self.deivate_loss + self.unobserved_pseudo_loss)

        opt_denoise = self.optimizer_func(self.propensity_model.parameters(), self.propensity_learning_rate)
        opt_ranker = self.optimizer_func(self.model.parameters(), self.learning_rate)

        opt_denoise.zero_grad()
        opt_ranker.zero_grad()

        self.loss.backward()

        if self.hparams.max_gradient_norm > 0:
            nn.utils.clip_grad_norm_(self.propensity_model.parameters(), self.hparams.max_gradient_norm)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)

        opt_denoise.step()
        opt_ranker.step()

        total_norm = 0

        for p in denoise_params:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        for p in ranking_model_params:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.norm = total_norm

    def softmax_cross_entropy_with_prob(self, outputs, labels):
        assert outputs.size(0) == labels.size(0)
        assert outputs.size(1) == labels.size(1)

        # [batch_size, rank_list_size]
        criterion = torch.nn.BCELoss()
        loss = criterion(outputs, labels)
        return loss

    def generate_pseudo_id_list(self, sample_num):
        # [rank_list_size, batch_size]
        sampled_pseudo_id_list = []

        for _ in range(sample_num):
            random_indices = np.random.choice(self.sample_cutoff, self.rank_list_size, replace=False)
            current_batch = self.docid_inputs[random_indices, :]
            sampled_pseudo_id_list.append(current_batch)

        # [rank_list_size, batch_size * sample_sum]
        sampled_pseudo_id_list = torch.cat(sampled_pseudo_id_list, 1)
        return sampled_pseudo_id_list

    def train_simulator(self, input_feed):
        """ Run a step of the simulator training, feeding the given inputs for training process.
        :param input_feed: (dictionary) A dictionary containing all the input feed data.
        :param teacher_forcing_ratio: the probability to apply the ground truth to train the model
        :return: A triple consisting of the loss, outputs (None if we do backward),
                 and a tf.summary containing related information about the step.
        """
        self.global_step += 1
        self.user_simulator.train()
        self.create_input_feed(input_feed, self.rank_list_size)

        train_output = self.ranking_model(self.user_simulator, self.rank_list_size,
                                          input_labels=self.labels)
        self.loss = self.env_loss_func(train_output, self.labels)
        self.opt_step(self.optimizer_simulator, self.user_simulator.parameters())

        nn.utils.clip_grad_value_(self.labels, 1)
        print(" [User Simulator] Loss %f at Global Step %d: " % (self.loss.item(), self.global_step))
        return self.loss.item(), None, self.train_summary

    def eval_simulator(self, input_feed, ranking_model=None):
        rank_list_size = 10

        self.user_simulator.eval()
        self.create_input_feed(input_feed, self.max_candidate_num)

        if ranking_model is None:
            # self.labels: [batch_size, max_doc]
            labels = self.labels.cpu()
            _, indices = labels.sort(descending=True, dim=-1)
            indices = indices[:, :rank_list_size]

            # self.docid_inputs: [max_doc, batch_size]
            docid_inputs = self.docid_inputs.transpose(0, 1).cpu()
            sorted_ids = torch.gather(docid_inputs, dim=1, index=indices)
            sorted_ids = sorted_ids.transpose(0, 1)

            with torch.no_grad():
                self.output = self.get_ranking_scores(model=self.user_simulator, input_id_list=sorted_ids, return_probs=True)

            # [batch, rank_list_size]
            output_scores = torch.cat(self.output, 1)
        else:
            with torch.no_grad():
                ranking_output = self.ranking_model(ranking_model.model, self.max_candidate_num)
                labels = ranking_output.cpu()
                _, indices = labels.sort(descending=True, dim=-1)
                indices = indices[:, :rank_list_size]

                # self.docid_inputs: [max_doc, batch_size]
                docid_inputs = self.docid_inputs.transpose(0, 1).cpu()
                sorted_ids = torch.gather(docid_inputs, dim=1, index=indices)
                sorted_ids = sorted_ids.transpose(0, 1)

                with torch.no_grad():
                    self.output = self.get_ranking_scores(model=self.user_simulator, input_id_list=sorted_ids,
                                                          return_probs=True)

                # [batch, rank_list_size]
                output_scores = torch.cat(self.output, 1)

        discounts = (torch.tensor(1) / torch.log2(torch.arange(output_scores.size(1), dtype=torch.float) + 2.0)).to(
            device=self.cuda)
        dcg_ac = (output_scores * discounts).sum(dim=-1).mean()
        eval_summary = {'dcg_ac': dcg_ac}
        return None, output_scores, eval_summary  # no loss, outputs, summary.

    def train(self, input_feed):
        """Run a step of the model feeding the given inputs.
        Args:
            input_feed: (dictionary) A dictionary containing all the input feed data.
        Returns:
            A triple consisting of the loss, outputs (None if we do backward),
            and a tf.summary containing related information about the step.
        """
        # Build model
        self.rank_list_size = self.exp_settings['selection_bias_cutoff']
        self.model.train()
        self.propensity_model.train()
        self.user_simulator.eval()

        self.create_input_feed(input_feed, self.max_candidate_num)
        train_output = self.ranking_model(self.model, self.rank_list_size)
        train_labels = self.labels[:, :self.rank_list_size]

        propensity_labels = torch.transpose(train_labels, 0, 1)
        self.propensity = self.propensity_model(propensity_labels)
        with torch.no_grad():
            self.propensity_weights = self.get_normalized_weights(self.logits_to_prob(self.propensity))
        self.rank_loss = self.loss_func(train_output, train_labels, self.propensity_weights)

        # Compute examination loss
        with torch.no_grad():
            self.relevance_weights = self.get_normalized_weights(self.logits_to_prob(train_output))
        self.exam_loss = self.loss_func(self.propensity, train_labels, self.relevance_weights)

        # IPS loss for observed ranking list with pseudo lists
        with torch.no_grad():
            observed_pseudo_labels = self.ranking_model(self.user_simulator, self.rank_list_size)
        self.observed_pseudo_loss = self.loss_func(train_output, observed_pseudo_labels, self.propensity_weights)
        self.deivate_loss = self.rank_loss - self.observed_pseudo_loss

        # direct loss on unobserved ranking lists
        pseudo_id_list = self.generate_pseudo_id_list(self.sample_num)
        with torch.no_grad():
            unobserved_pseudo_labels = self.get_ranking_scores(model=self.user_simulator, input_id_list=pseudo_id_list)
        unobserved_pseudo_output = self.get_ranking_scores(model=self.model, input_id_list=pseudo_id_list)
        unobserved_pseudo_labels = torch.cat(unobserved_pseudo_labels, 1)
        unobserved_pseudo_output = torch.cat(unobserved_pseudo_output, 1)
        self.unobserved_pseudo_loss = self.loss_func(unobserved_pseudo_output, unobserved_pseudo_labels)

        # Gradients and SGD update operation for training the model.
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * (self.deivate_loss + self.unobserved_pseudo_loss)
        self.separate_gradient_update()

        self.clip_grad_value(train_labels, clip_value_min=0, clip_value_max=1)
        self.clip_grad_value(observed_pseudo_labels, clip_value_min=0, clip_value_max=1)
        self.clip_grad_value(unobserved_pseudo_labels, clip_value_min=0, clip_value_max=1)
        print(" [Ranking Model] Loss %f at Global Step %d: " % (self.loss.item(), self.global_step))
        self.global_step += 1
        return self.loss.item(), None, self.train_summary

    def validation(self, input_feed, is_online_simulation=False):
        self.model.eval()
        self.create_input_feed(input_feed, self.max_candidate_num)
        with torch.no_grad():
            self.output = self.ranking_model(self.model,
                                             self.max_candidate_num)
        if not is_online_simulation:
            pad_removed_output = self.remove_padding_for_metric_eval(
                self.docid_inputs, self.output)
            # reshape from [max_candidate_num, ?] to [?, max_candidate_num]
            for metric in self.exp_settings['metrics']:
                topn = self.exp_settings['metrics_topn']
                metric_values = ultra.utils.make_ranking_metric_fn(
                    metric, topn)(self.labels, pad_removed_output, None)
                for topn, metric_value in zip(topn, metric_values):
                    self.create_summary('%s_%d' % (metric, topn),
                                        '%s_%d' % (metric, topn), metric_value.item(), False)
        return None, self.output, self.eval_summary # no loss, outputs, summary.

    def get_normalized_weights(self, propensity):
        """Computes listwise softmax loss with propensity weighting.
        Args:
            propensity: (tf.Tensor) A tensor of the same shape as `output` containing the weight of each element.
        Returns:
            (tf.Tensor) A tensor containing the propensity weights.
        """
        propensity_list = torch.unbind(
            propensity, dim=1)  # Compute propensity weights
        pw_list = []
        for i in range(len(propensity_list)):
            pw_i = propensity_list[0] / propensity_list[i]
            pw_list.append(pw_i)
        propensity_weights = torch.stack(pw_list, dim=1)
        if self.hparams.max_propensity_weight > 0:
            self.clip_grad_value(propensity_weights,clip_value_min=0,
                clip_value_max=self.hparams.max_propensity_weight)
        return propensity_weights

    def clip_grad_value(self, parameters, clip_value_min, clip_value_max) -> None:
        r"""Clips gradient of an iterable of parameters at specified value.
        Gradients are modified in-place.
        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            clip_value (float or int): maximum allowed value of the gradients.
                The gradients are clipped in the range
                :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        clip_value_min = float(clip_value_min)
        clip_value_max = float(clip_value_max)
        for p in filter(lambda p: p.grad is not None, parameters):
            p.grad.data.clamp_(min=clip_value_min, max=clip_value_max)