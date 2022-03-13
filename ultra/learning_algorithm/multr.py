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
import torchsnooper


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
                              num_layers=num_layers, bidirectional=True, batch_first=False)
        self.decoder = nn.GRU(input_size=hidden_size * 4, hidden_size=hidden_size,
                              num_layers=num_layers, bidirectional=False, batch_first=False)
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(100, hidden_size)
        self.click_embedding = nn.Embedding(2, self.hidden_size)

        self.proj_query = nn.Linear(hidden_size * 2, hidden_size)
        self.proj_result = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, 2)

    def convert_prob_to_click(self, input_labels):
        """
        :param input_labels: [rank_list_size, batch_size]
        :return: tensor: [rank_list_size, batch_size]
        """
        rank_list_size, batch_size = input_labels.size(0), input_labels.size(1)

        input_clicks = []
        for i in range(batch_size):
            while True:
                click_prob = input_labels[:, i].unsqueeze(1)
                non_click_prob = 1.0 - click_prob
                weights = torch.cat([non_click_prob, click_prob], 1)
                clicks = torch.multinomial(weights, 1, replacement=True)
                if clicks.sum() > 0:
                    input_clicks.append(clicks)
                    break
        input_clicks = torch.stack(input_clicks, 1).squeeze(-1)
        return input_clicks

    def sample(self, click_logits):
        """
        :param click_logits: [batch_size, 2]
        :return: sampled clicks [batch_size, 1]
        """
        samples = F.gumbel_softmax(click_logits, hard=True)
        clicks = torch.argmax(samples, -1, keepdim=True)
        return clicks

    def predict_with_label(self, input_data, input_labels, teacher_forcing_ratio):
        """ infer the conditional click probability for doc at each position
        :param input_data: [rank_list_size, batch_size, feature_size]
        :param input_labels: [rank_list_size, batch_size]
        :return:
        """
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        rank_list_size, batch_size = input_data.size(0), input_data.size(1)
        input_tensor = self.input_transform(input_data)

        # query feature [rank_list_size, batch_size, D]
        output, hidden = self.encoder(input_tensor)
        query_context = output[-1].unsqueeze(0)
        query_feature = self.proj_query(query_context).expand(rank_list_size, -1, -1)

        # doc features [rank_list_size, batch_size, D]
        result_feature = self.proj_result(input_tensor)

        # positions feature [rank_list_size, batch_size, D]
        position_indices = torch.arange(rank_list_size, dtype=torch.long).to(input_data.device)
        position_feature = self.position_embedding(position_indices)
        position_feature = position_feature.unsqueeze(1).expand(-1, batch_size, -1)

        if use_teacher_forcing:
            # [rank_list_size, batch_size]
            # input_labels = self.convert_prob_to_click(input_labels)
            input_clicks = input_labels.long()
            padding = Variable(torch.zeros(1, batch_size)).long().to(input_data.device)
            input_clicks = torch.cat((padding, input_clicks), dim=0)
            input_clicks = input_clicks[:-1, :]
            # [rank_list_size, batch_size, D]
            click_feature = self.click_embedding(input_clicks)

            input_feature = torch.cat((query_feature, result_feature, position_feature, click_feature), 2)
            init_state = Variable(torch.zeros((1 * self.num_layers, batch_size, self.hidden_size))).to(
                input_data.device)
            output, hidden = self.decoder(input_feature, init_state)
            output = self.dropout(output)
            # [rank_list_size, batch_size, 2]
            output_logits = self.output_linear(output)
            # [rank_list_size, batch_size, 1, 2]
            return output_logits.unsqueeze(2)

        else:
            output_logits = []
            decoder_hidden = Variable(torch.zeros((1 * self.num_layers, batch_size, self.hidden_size))).to(
                input_data.device)
            previous_click = Variable(torch.zeros(batch_size, 1)).long().to(input_data.device)
            for i in range(rank_list_size):
                # [batch_size, 1, D]
                click_feature = self.click_embedding(previous_click)
                click_feature = torch.squeeze(click_feature, 1)
                input_feature = torch.cat((query_feature[i], result_feature[i], position_feature[i], click_feature), -1)
                # [1, batch_size, D * 4]
                input_feature = torch.unsqueeze(input_feature, 0)

                decoder_output, decoder_hidden = self.decoder(input_feature, decoder_hidden)
                decoder_output = self.dropout(decoder_output)
                # [1, batch_size, 2]
                decoder_logits = self.output_linear(decoder_output)
                click_logits = torch.squeeze(decoder_logits, 0)

                # [batch_size, 2]
                output_logits.append(click_logits)
                previous_click = self.sample(click_logits)

            # [rank_list_size, batch_size, 2]
            output_logits = torch.stack(output_logits, 0)
            # [rank_list_size, batch_size, 1, 2]
            output_logits = torch.unsqueeze(output_logits, 2)
            return output_logits

    def predict_without_label(self, input_data):
        """ infer clicks for doc at each position
        :param input_data:
        :return:
        """
        rank_list_size, batch_size = input_data.size(0), input_data.size(1)
        input_tensor = self.input_transform(input_data)

        # query feature [rank_list_size, batch_size, D]
        output, hidden = self.encoder(input_tensor)
        query_context = output[-1].unsqueeze(0)
        query_feature = self.proj_query(query_context).expand(rank_list_size, -1, -1)

        # doc features [rank_list_size, batch_size, D]
        result_feature = self.proj_result(input_tensor)

        # positions feature [rank_list_size, batch_size, D]
        position_indices = torch.arange(rank_list_size, dtype=torch.long).to(input_data.device)
        position_feature = self.position_embedding(position_indices)
        position_feature = position_feature.unsqueeze(1).expand(-1, batch_size, -1)

        output_probs, output_clicks = [], []
        decoder_hidden = Variable(torch.zeros((1 * self.num_layers, batch_size, self.hidden_size))).to(input_data.device)
        previous_click = Variable(torch.zeros(batch_size, 1)).long().to(input_data.device)
        for i in range(rank_list_size):
            # [batch_size, 1, D]
            click_feature = self.click_embedding(previous_click)
            click_feature = torch.squeeze(click_feature, 1)
            input_feature = torch.cat((query_feature[i], result_feature[i], position_feature[i], click_feature), -1)
            # [1, batch_size, D * 4]
            input_feature = torch.unsqueeze(input_feature, 0)

            decoder_output, decoder_hidden = self.decoder(input_feature, decoder_hidden)
            decoder_output = self.dropout(decoder_output)
            # [1, batch_size, 2]
            decoder_logits = self.output_linear(decoder_output)
            click_logits = torch.squeeze(decoder_logits, 0)

            # [batch_size, 1]
            click_now = self.sample(click_logits)
            # [batch_size, 1]
            click_probs = F.softmax(click_logits, -1)[:, 1].unsqueeze(-1)

            previous_click = click_now
            output_clicks.append(click_now)
            output_probs.append(click_probs)

        # [rank_list_size, batch_size, 1]
        output_clicks = torch.stack(output_clicks, 0)
        # [rank_list_size, batch_size, 1]
        output_probs = torch.stack(output_probs, 0)
        return output_clicks, output_probs

    def build(self, input_list, input_labels=None, return_clicks=True, teacher_forcing_ratio=.0):
        """ input_list --> input_data [rank_list_size, batch_size, D]
            input_labels ---> [batch_size, rank_list_size]
        """
        device = next(self.parameters()).device
        # [rank_list_size, batch_size, D]
        input_data = torch.stack(input_list, dim=0).to(dtype=torch.float32, device=device)
        if input_labels is None:
            output_clicks, output_probs = self.predict_without_label(input_data)
            if return_clicks:
                # [rank_list_size, batch_size, 1]
                return torch.unbind(output_clicks, dim=0)
            else:
                # [rank_list_size, batch_size, 1]
                return torch.unbind(output_probs, dim=0)
        else:
            # [rank_list_size, batch_size]
            input_labels = input_labels.transpose(0, 1)
            # [rank_list_size, batch_size, 1, 2]
            output_logits = self.predict_with_label(input_data, input_labels, teacher_forcing_ratio)
            return torch.unbind(output_logits, dim=0)


class MULTR(BaseAlgorithm):
    def __init__(self, data_set, exp_settings, forward_only=False):
        print('Build Model-based Unbiased Learning to Rank Model')

        self.hparams = ultra.utils.hparams.HParams(
            learning_rate=0.5,                    # Learning rate
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
            constant_propensity_initialization=False,                   # Set true to initialize propensity with constants.
            env_loss_func='softmax_cross_entropy_with_logit',           # Select Loss function
            sample_num=24,                        #
            hidden_size=64,
            teacher_forcing_ratio=0.5
        )

        print(exp_settings['learning_algorithm_hparams'])
        self.cuda = torch.device('cuda')
        self.is_cuda_avail = torch.cuda.is_available()
        self.writer = SummaryWriter()
        self.train_summary = {}
        self.eval_summary = {}
        self.hparams.parse(exp_settings['learning_algorithm_hparams'])
        self.exp_settings = exp_settings
        self.max_candidate_num = exp_settings['max_candidate_num']
        self.feature_size = data_set.feature_size
        if 'selection_bias_cutoff' in exp_settings.keys():
            self.rank_list_size = self.exp_settings['selection_bias_cutoff']
            self.propensity_model = DenoisingNet(self.rank_list_size)
        self.model = self.create_model(self.feature_size)
        self.user_simulator = UserSimulator(self.feature_size, self.hparams.hidden_size, num_layers=1, dropout=0.4)

        if self.is_cuda_avail:
            self.model = self.model.to(device=self.cuda)
            self.propensity_model = self.propensity_model.to(device=self.cuda)
            self.user_simulator = self.user_simulator.to(device=self.cuda)

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
        self.env_learning_rate = float(self.hparams.env_learning_rate)

        self.global_step = 0

        # Select logits to prob function
        self.logits_to_prob = nn.Softmax(dim=-1)
        if self.hparams.logits_to_prob == 'sigmoid':
            self.logits_to_prob = sigmoid_prob

        self.optimizer_func = torch.optim.Adagrad
        if self.hparams.grad_strategy == 'sgd':
            self.optimizer_func = torch.optim.SGD
        self.optimizer_simulator = torch.optim.Adam(self.user_simulator.parameters(),
                                                    lr=self.hparams.env_learning_rate,
                                                    weight_decay=self.hparams.env_l2_loss)

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
        if self.hparams.env_loss_func == 'softmax_cross_entropy_with_logit':
            self.env_loss_func = self.softmax_cross_entropy_with_logit
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
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

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

    def softmax_cross_entropy_with_logit(self, outputs, labels):
        """
        :param outputs: [batch_size, rank_list_size, 2]
        :param labels:  [batch_size, rank_list_size]
        :return:
        """
        assert outputs.size(0) == labels.size(0)
        assert outputs.size(1) == labels.size(1)

        batch_size, rank_list_size = labels.size(0), labels.size(1)
        outputs = outputs.contiguous().view(-1, 2)
        labels = labels.contiguous().view(-1).long()

        # [batch_size, rank_list_size]
        loss = torch.nn.CrossEntropyLoss(reduction="none")(outputs, labels)
        loss = torch.sum(loss) / batch_size
        return loss

    def generate_pseudo_id_list(self, sample_num):
        # [rank_list_size, batch_size]
        sampled_pseudo_id_list = []

        for _ in range(sample_num):
            random_indices = np.random.choice(self.rank_list_size, self.rank_list_size, replace=False)
            current_batch = self.docid_inputs[random_indices, :]
            sampled_pseudo_id_list.append(current_batch)

        # [rank_list_size, batch_size * sample_sum]
        sampled_pseudo_id_list = torch.cat(sampled_pseudo_id_list, 1)
        return sampled_pseudo_id_list

    def train_simulator(self, input_feed):
        """ Run a step of the simulator training, feeding the given inputs for training process.
        :param input_feed: (dictionary) A dictionary containing all the input feed data.
        :return: A triple consisting of the loss, outputs (None if we do backward),
                 and a tf.summary containing related information about the step.
        """
        self.user_simulator.train()
        self.create_input_feed(input_feed, self.rank_list_size)

        train_output = self.ranking_model(self.user_simulator, self.rank_list_size, input_labels=self.labels,
                                          teacher_forcing_ratio=self.hparams.teacher_forcing_ratio)

        self.loss = self.env_loss_func(train_output, self.labels)
        self.opt_step(self.optimizer_simulator, self.user_simulator.parameters())

        nn.utils.clip_grad_value_(self.labels, 1)
        print(" [User Simulator] Loss %f at Global Step %d: " % (self.loss.item(), self.global_step))
        self.global_step += 1
        return self.loss.item(), None, self.train_summary

    def eval_simulator(self, input_feed):
        self.user_simulator.eval()
        self.create_input_feed(input_feed, self.rank_list_size)

        with torch.no_grad():
            test_output_probs = self.ranking_model(self.user_simulator, self.rank_list_size,
                                                   return_clicks=False)
        prob_delta = torch.abs(self.labels - test_output_probs).mean()
        eval_summary = {'prob_abs': prob_delta}
        return None, test_output_probs, eval_summary

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

        self.create_input_feed(input_feed, self.rank_list_size)
        train_output = self.ranking_model(self.model, self.rank_list_size)

        propensity_labels = torch.transpose(self.labels, 0, 1)
        self.propensity = self.propensity_model(propensity_labels)
        with torch.no_grad():
            self.propensity_weights = self.get_normalized_weights(self.logits_to_prob(self.propensity))
        self.rank_loss = self.loss_func(train_output, self.labels, self.propensity_weights)

        # Compute examination loss
        with torch.no_grad():
            self.relevance_weights = self.get_normalized_weights(self.logits_to_prob(train_output))
        self.exam_loss = self.loss_func(self.propensity, self.labels, self.relevance_weights)

        # IPS loss for observed ranking list with pseudo lists
        with torch.no_grad():
            observed_pseudo_labels = self.ranking_model(self.user_simulator, self.rank_list_size)
        self.observed_pseudo_loss = self.loss_func(train_output, observed_pseudo_labels, self.propensity_weights)
        self.deivate_loss = self.rank_loss - self.observed_pseudo_loss

        # direct loss on unobserved ranking lists
        pseudo_id_list = self.generate_pseudo_id_list(self.hparams.sample_num)
        with torch.no_grad():
            unobserved_pseudo_labels = self.get_ranking_scores(model=self.user_simulator, input_id_list=pseudo_id_list)
        unobserved_pseudo_output = self.get_ranking_scores(model=self.model, input_id_list=pseudo_id_list)
        unobserved_pseudo_labels = torch.cat(unobserved_pseudo_labels, 1)
        unobserved_pseudo_output = torch.cat(unobserved_pseudo_output, 1)
        self.unobserved_pseudo_loss = self.loss_func(unobserved_pseudo_output, unobserved_pseudo_labels)

        # Gradients and SGD update operation for training the model.
        self.loss = self.exam_loss + self.hparams.ranker_loss_weight * (self.deivate_loss + self.unobserved_pseudo_loss)
        self.separate_gradient_update()

        self.clip_grad_value(self.labels, clip_value_min=0, clip_value_max=1)
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
            self.output = self.output - torch.min(self.output)

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
            pw_i = propensity_list[0] / (propensity_list[i] + 1e-6)
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