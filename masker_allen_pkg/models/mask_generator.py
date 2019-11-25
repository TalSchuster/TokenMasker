from typing import Dict, List, Tuple, Union, Any

import os
import copy
import torch
from torch.nn.functional import gumbel_softmax
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.models.archival import load_archive
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy, Average
from masker_allen_pkg.training.metrics import AccuracyVSDeletion, F1SequenceMeasure
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention

CONFIG_NAME = "config.json"


@Model.register('mask_generator')
class MaskGenerator(Model):
    """
    Use the classifier and nmask predictor models to generate masks to neutralize the premise.

    Important - share the vocabulary between models by using vocabulary: {directory_path: ...}
    in the configuration.

    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 contextualizer: Seq2SeqEncoder,
                 labeler: Seq2SeqEncoder,
                 projection_size: int,
                 bidirectional: bool = False,
                 use_hypothesis: bool = True,
                 attention: str = "", # "" - none / cosine / bilinear
                 initializer: InitializerApplicator = None,
                 classifier_dir = "",
                 del_perc_lambda = 1,
                 del_perc = 0.3,
                 del_metric_threshold = 0.1,
                 teacher_lambda = 0.0,
                 coverage_lambda = 0.0,
                 transition_lamb = 0.0,
                 gumbel = True,
                 neutral_label = "") -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder

        if contextualizer.is_bidirectional() is not bidirectional:
            raise ConfigurationError(
                    "Bidirectionality of contextualizer must match bidirectionality of "
                    "language model. "
                    f"Contextualizer bidirectional: {contextualizer.is_bidirectional()}, "
                    f"language model bidirectional: {bidirectional}")

        self.classifier_dir = classifier_dir
        self.classifier = None
        self.coverage_lambda = coverage_lambda
        self.del_perc_lambda = del_perc_lambda
        self.del_perc = del_perc
        self.teacher_lambda = teacher_lambda
        self.transition_lamb = transition_lamb
        self.gumbel = gumbel
        if classifier_dir != "":
            overrides = '{"model": {"dropout": 0, "output_feedforward": {"dropout": 0}}}'
            overrides = ""
            archive = load_archive(classifier_dir, overrides=overrides)

            self.classifier = archive.model
            # Freeze parameters
            for p in self.classifier.parameters():
                p.requires_grad = False

            # A hack that prevents allennlp from crushing when running extend on all submodules
            def foo(*x, **y): return 1
            self.classifier._text_field_embedder.token_embedder_tokens.extend_vocab = foo
            self.classifier.eval()

            # get index of the neutral label
            self.neutral_ind = self.classifier.vocab.get_token_index(neutral_label, 'labels')

        self.criterion = torch.nn.CrossEntropyLoss()

        self._contextualizer = contextualizer
        self._labeler = labeler
        self._bidirectional = bidirectional
        self.use_hypothesis = use_hypothesis
        self.attention = attention
        self.projection_size = projection_size

        # hypothesis aggr
        self.w_prem = torch.nn.Linear(contextualizer.get_output_dim(), projection_size)
        if use_hypothesis:
            self.w_hyp = torch.nn.Linear(contextualizer.get_output_dim(), projection_size)

        self._contextual_dim = contextualizer.get_output_dim()
        # The dimension for making predictions just in the forward
        # (or backward) direction.
        if self._bidirectional:
            self._forward_dim = self._contextual_dim // 2
        else:
            self._forward_dim = self._contextual_dim

        if self.attention:
            if self.attention == "cosine":
                self.attention_mat = CosineMatrixAttention()
            elif self.attention == "bilinear":
                self.attention_mat = BilinearMatrixAttention(self._forward_dim, self._forward_dim)
            else:
                raise ConfigurationError("Undefined attention type")

        self.mask_linear = torch.nn.Linear(self._labeler.get_output_dim(), 2)

        self._accuracy = CategoricalAccuracy()
        self._avg_perc_masked = Average()
        self._avg_transition = Average()
        self._acc_vs_del = AccuracyVSDeletion(del_threshold=del_metric_threshold)
        self._acc_plus_del = AccuracyVSDeletion(del_threshold=0, aggr="sum")
        self._f1_deletions = F1SequenceMeasure(positive_label=1)
        if initializer is not None:
            initializer(self)

    def cuda(self, device=None):
        '''
        override cuda function to move submodules also to the same device
        '''
        if self.classifier:
            self.classifier.cuda(device)

        return super(MaskGenerator, self).cuda(device)


    def _get_target_token_embeddings(self,
                                     token_embeddings: torch.Tensor,
                                     mask: torch.Tensor,
                                     direction: int) -> torch.Tensor:
        # Need to shift the mask in the correct direction
        zero_col = token_embeddings.new_zeros(mask.size(0), 1).byte()
        if direction == 0:
            # forward direction, get token to right
            shifted_mask = torch.cat([zero_col, mask[:, 0:-1]], dim=1)
        else:
            shifted_mask = torch.cat([mask[:, 1:], zero_col], dim=1)
        return token_embeddings.masked_select(shifted_mask.unsqueeze(-1)).view(-1, self._contextual_dim)

    def num_layers(self) -> int:
        """
        Returns the depth of this LM. That is, how many layers the contextualizer has plus one for
        the non-contextual layer.
        """
        if hasattr(self._contextualizer, 'num_layers'):
            return self._contextualizer.num_layers + 1
        else:
            raise NotImplementedError(f"Contextualizer of type {type(self._contextualizer)} " +
                                      "does not report how many layers it has.")

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]] = None,
                label: torch.IntTensor = None,
                premise_delete: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        Computes the averaged forward (and backward, if language model is bidirectional)
        LM loss from the batch.

        By convention, the input dict is required to have at least a ``"tokens"``
        entry that's the output of a ``SingleIdTokenIndexer``, which is used
        to compute the language model targets.

        Parameters
        ----------
        tokens: ``torch.Tensor``, required.
            The output of ``Batch.as_tensor_dict()`` for a batch of sentences.
        premise_delete: the mask to use for the input

        Returns
        -------
        Dict with keys:

        ``'loss'``: ``torch.Tensor``
            forward negative log likelihood, or the average of forward/backward
            if language model is bidirectional
        ``'forward_loss'``: ``torch.Tensor``
            forward direction negative log likelihood
        ``'backward_loss'``: ``torch.Tensor`` or ``None``
            backward direction negative log likelihood. If language model is not
            bidirectional, this is ``None``.
        ``'lm_embeddings'``: ``Union[torch.Tensor, List[torch.Tensor]]``
            (batch_size, timesteps, embed_dim) tensor of top layer contextual representations or
            list of all layers. No dropout applied.
        ``'noncontextual_token_embeddings'``: ``torch.Tensor``
            (batch_size, timesteps, token_embed_dim) tensor of bottom layer noncontextual
            representations
        ``'mask'``: ``torch.Tensor``
            (batch_size, timesteps) mask for the embeddings
        """
        # pylint: disable=arguments-differ
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()
        #mask = get_text_field_mask(source)

        # shape (batch_size, timesteps, embedding_size)
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)

        # get mask for premise_delete (the sentences that have masked_inds values in data)
        first_inds = premise_delete[:,0]
        premise_delete_mask = (first_inds != -1).long()

        contextual_premise: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
                embedded_premise, premise_mask
        )

        # shape (batch size, sent_len, w_hyp_output)
        proj_embedded_premise = self.w_prem(contextual_premise)

        if self.use_hypothesis:
            contextual_hyp: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
                    embedded_hypothesis, hypothesis_mask
            )
            proj_embedded_hypothesis = self.w_hyp(contextual_hyp)
            masked_embedded_hypothesis = proj_embedded_hypothesis * hypothesis_mask.unsqueeze(-1)
            words_per_hyp_sent = hypothesis_mask.sum(-1)

            if self.attention:
                # Shape: (batch_size, premise_length, hypothesis_length)
                similarity_mat = self.attention_mat(proj_embedded_premise, proj_embedded_hypothesis)
                # Shape: (batch_size, premise_length, hypothesis_length)
                p2h_attention = masked_softmax(similarity_mat, hypothesis_mask)
                # Shape: (batch_size, premise_length, projection_dim)
                aggr_premise = weighted_sum(proj_embedded_hypothesis, p2h_attention) + proj_embedded_premise
            else:
                # shape (batch size, w_hyp_output)
                mean_hyp = torch.sum(masked_embedded_hypothesis,dim=1) / words_per_hyp_sent.unsqueeze(-1)

                mean_hyp = mean_hyp[:,None,:].expand_as(proj_embedded_premise)
                aggr_premise = proj_embedded_premise + mean_hyp


            aggr_premise: Union[torch.Tensor, List[torch.Tensor]] = self._labeler(
                    aggr_premise, premise_mask
            )
        else:
            aggr_premise = proj_embedded_premise

        return_dict = {}

        token_ids = premise.get("tokens")
        assert isinstance(aggr_premise, torch.Tensor)

        # shape batch_size * sent_len * 2
        delete_logits = self.mask_linear(aggr_premise)

        if self.gumbel:
            delete_probs = gumbel_softmax(delete_logits.view(-1,2),hard=True).view_as(delete_logits)
        else:
            # shape batch_size * 2 * sent_len
            delete_logits.transpose_(-1,-2)
            delete_probs = masked_softmax(delete_logits, premise_mask, dim=1)

            # shape batch_size * sent_len * 2
            delete_probs.transpose_(-1,-2)

        gen_premise_delete = delete_probs[:,:,1]

        gen_premise_delete = gen_premise_delete * premise_mask

        # for regular (not gumbel) softmax
        if not self.training:
            gen_premise_delete = (gen_premise_delete > 0.5).float()

        if self.classifier:
            target_label = label.new_ones(label.shape) * self.neutral_ind
            self.classifier._modules['rnn_input_dropout'].eval()
            self.classifier._modules['dropout'].eval()
            self.classifier._modules['_output_feedforward']._modules['_dropout'].eval()
            out = self.classifier.forward(premise, hypothesis, target_label, metadata, gen_premise_delete)

            target_prob = out['label_probs']
            loss = self.criterion(target_prob, target_label)

        # regularization from given premise_delete loss
        teacher_available = premise_delete_mask.nonzero()
        if len(teacher_available) > 0:
            premise_delete_masked = premise_delete[teacher_available].squeeze()
            premise_del_dist = torch.dist(gen_premise_delete[teacher_available].squeeze(), premise_delete_masked)
            loss += self.teacher_lambda * premise_del_dist / len(teacher_available)

            self._f1_deletions(delete_probs[teacher_available], premise_delete_masked, premise_mask[teacher_available].squeeze())

        self._accuracy(target_prob, target_label)
        delete_size = gen_premise_delete.sum(1) / premise_mask.sum(1)

        # delete percentage loss
        del_perc_loss = (delete_size.mean() - self.del_perc)**2
        loss += self.del_perc_lambda * del_perc_loss

        # coverage loss (should probably not use del_perc with this)
        loss += self.coverage_lambda * delete_size.mean()

        # Transition loss:
        rolled = torch.cat((gen_premise_delete[:,1:], gen_premise_delete[:,-1:]), dim=-1)
        transitions = torch.abs(rolled - gen_premise_delete)
        premise_mask_rolled = torch.cat((premise_mask[:,1:], premise_mask.new_zeros(premise_mask[:,-1:].shape)), dim=-1)
        transitions = transitions * premise_mask_rolled
        mean_transitions = transitions.sum(1).mean()
        loss += self.transition_lamb * mean_transitions

        # the round is important when not using gumbel (soft deletions)
        gen_premise_delete_round = (gen_premise_delete > 0.5).float()
        delete_size_round = gen_premise_delete_round.sum(1) / premise_mask.sum(1)
        rolled_round = torch.cat((gen_premise_delete_round[:,1:], gen_premise_delete_round[:,-1:]), dim=-1)
        transitions_round = torch.abs(rolled_round - gen_premise_delete_round)
        transitions_round = transitions_round * premise_mask_rolled

        self._avg_perc_masked(delete_size_round.mean().item())
        self._avg_transition(transitions_round.sum(1).mean().item())
        self._acc_vs_del(self._accuracy.get_metric(), self._avg_perc_masked.get_metric())
        self._acc_plus_del(self._accuracy.get_metric(), self._avg_perc_masked.get_metric())
        out.update({
                'loss': loss,
                'deleted_inds': gen_premise_delete,
            })

        return out

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self._f1_deletions.get_metric(reset)
        return {
                'accuracy': self._accuracy.get_metric(reset),
                'delete_size': self._avg_perc_masked.get_metric(reset),
                'transitions': self._avg_transition.get_metric(reset),
                'acc_vs_del': self._acc_vs_del.get_metric(reset),
                'acc_plus_del': self._acc_plus_del.get_metric(reset),
                'm_prec': precision,
                'm_recall': recall,
                'm_f1': f1_measure,
                }
