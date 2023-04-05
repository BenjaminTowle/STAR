import abc
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.configuration_utils import PretrainedConfig
from transformers import ( 
    BertModel,
    BertPreTrainedModel, 
    DistilBertModel,
    DistilBertPreTrainedModel,
    T5Model,
    T5PreTrainedModel,
    T5EncoderModel,
)
from typing import Optional

from src.modeling.losses import get_loss_fn
from src.modeling.utils import get_attention_mask

from torch.nn import BatchNorm1d

@dataclass
class MatchingOutput(SequenceClassifierOutput):
    x_embeds: Optional[torch.Tensor] = None
    y_embeds: Optional[torch.Tensor] = None


class MatchingMixin(abc.ABC):
    """
    A Mixin that performs biencoder-like functions for an arbitrary embedding model.
    """
    subclasses = {}  # Mapping of strs to model subclasses

    def __init__(self, config: PretrainedConfig, use_symmetric_loss: bool) -> None:
        super().__init__(config)
        self._loss_fn = get_loss_fn(use_symmetric_loss)

    @abc.abstractmethod
    def get_embedding(self, input_ids, attention_mask):
        pass

    def forward_embedding(self, input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        device = input_ids.device
        attn_mask = torch.ones(input_ids.size()).long().to(device).masked_fill(input_ids == pad_token_id, 0)
        embeds = self.get_embedding(input_ids, attention_mask=attn_mask)
        return embeds

    def forward_biencoder(
        self,
        input_ids: torch.Tensor,
        y_input_ids: Optional[torch.Tensor] = None,
        candidate_input_ids: Optional[torch.Tensor] = None,
        candidate_embeds: Optional[torch.Tensor] = None
    ) -> MatchingOutput:

        x_embeds = self.get_embedding(input_ids, attention_mask=get_attention_mask(input_ids))

        if candidate_embeds is not None:
            y_embeds = candidate_embeds
        else:
            if candidate_input_ids is not None:
                y_input_ids = candidate_input_ids
            old_shape = y_input_ids.size() # Either bsz x len or bsz x k x len
            y_embeds = self.get_embedding(y_input_ids.reshape([-1, old_shape[-1]]), attention_mask=get_attention_mask(y_input_ids.reshape([-1, old_shape[-1]])))
            y_embeds = y_embeds.reshape(list(old_shape[:-1]) + [-1])

        return MatchingOutput(
            x_embeds=x_embeds,
            y_embeds=y_embeds
        )

    def loss_fn(
        self,
        x_embeds: torch.Tensor,
        y_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> SequenceClassifierOutput:

        scores = None
        if y_embeds.ndim == 2:
            loss = self._loss_fn(x_embeds, y_embeds)

        else:
            scores = (x_embeds.unsqueeze(1) * y_embeds).sum(-1)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(scores, labels)
        
        return SequenceClassifierOutput(
            logits=scores,
            loss=loss
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            y_input_ids: Optional[torch.Tensor] = None,
            candidate_input_ids: Optional[torch.Tensor] = None,
            candidate_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:

        outputs = self.forward_biencoder(
            input_ids=input_ids,
            y_input_ids=y_input_ids,
            candidate_input_ids=candidate_input_ids,
            candidate_embeds=candidate_embeds,
        )

        outputs = self.loss_fn(
            x_embeds=outputs.x_embeds,
            y_embeds=outputs.y_embeds,
            labels=labels
        )

        return outputs

    @classmethod
    def register_subclass(cls, model_type):
        def decorator(subclass):
            cls.subclasses[model_type] = subclass
            return subclass
        
        return decorator

    @classmethod
    def create(cls, model_type: str, model_path: str, *args, **kwargs):
        if model_type not in cls.subclasses:
            raise ValueError(f"Bad model_type: {model_type}.")
        
        return cls.subclasses[model_type].from_pretrained(model_path, *args, **kwargs)


@MatchingMixin.register_subclass("bert")
class BertMatchingModel(MatchingMixin, BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, use_symmetric_loss: bool = True):
        super().__init__(config, use_symmetric_loss)
        self.bert = BertModel(config)

    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]


@MatchingMixin.register_subclass("distilbert")
class DistilBertMatchingModel(MatchingMixin, DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, use_symmetric_loss: bool = True):
        super().__init__(config, use_symmetric_loss)
        self.distilbert = DistilBertModel(config)

    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.distilbert(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]


@MatchingMixin.register_subclass("t5")
class T5MatchingModel(MatchingMixin, T5Model):
    """An encoder-only T5 model that returns the embedding of the first token in the sequence."""
    def __init__(self, config: PretrainedConfig, use_symmetric_loss: bool = True):
        super().__init__(config, use_symmetric_loss)

    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.get_encoder()(input_ids, attention_mask=attention_mask).last_hidden_state.mean(1)


@MatchingMixin.register_subclass("similarity")
class DistilBertSimilarityModel(DistilBertMatchingModel):
    def __init__(self, config: PretrainedConfig, use_symmetric_loss: bool = True):
        super().__init__(config, use_symmetric_loss)
        #self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.classifier = nn.Linear(config.dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        y_input_ids: Optional[torch.Tensor] = None,
        candidate_input_ids: Optional[torch.Tensor] = None,
        candidate_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:

        outputs = self.forward_biencoder(
            input_ids=input_ids,
            y_input_ids=y_input_ids,
            candidate_input_ids=candidate_input_ids,
            candidate_embeds=candidate_embeds,
        )

        x_embeds = outputs.x_embeds
        y_embeds = outputs.y_embeds

        #x_embeds = F.relu(x_embeds)
        #y_embeds = F.relu(y_embeds)

        # Normalise embeds
        #torchx_embeds = x_embeds / x_embeds.norm(dim=-1, keepdim=True)
        #torchy_embeds = y_embeds / y_embeds.norm(dim=-1, keepdim=True)

        scores = self.classifier(x_embeds * y_embeds).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(scores, labels.float())  # This includes the sigmoid

        return SequenceClassifierOutput(
            logits=nn.Sigmoid()(scores),
            loss=loss
        )


#@MatchingMixin.register_subclass("distillation")
class DistilledMatchingModel(DistilBertMatchingModel):

    def forward(
        self,
        input_ids,
        z_input_ids,
        y_input_ids,
    ):

        # input_ids should comprise 3 positives
        assert z_input_ids.ndim == 3

        bsz, K, w = z_input_ids.size()

        z_input_ids = z_input_ids.reshape([bsz * 3, w])
        x_embeds = self.get_embedding(input_ids, attention_mask=get_attention_mask(input_ids))
        y_embeds = self.get_embedding(y_input_ids, attention_mask=get_attention_mask(y_input_ids))
        z_embeds = self.get_embedding(z_input_ids, attention_mask=get_attention_mask(z_input_ids))
        z_embeds = z_embeds.reshape([bsz, 3, -1])

        total_loss = 0
        for k in range(K):
            outputs = self.loss_fn(x_embeds, z_embeds[:, k, :])
            total_loss += outputs.loss
        outputs.loss = total_loss / K

        #outputs = self.loss_fn(x_embeds, y_embeds)
        #probs = F.softmax(logits, dim=-1)
        #kl_loss = (-1 * probs * torch.log_softmax((x_embeds.unsqueeze(1) * z_embeds).sum(-1), dim=-1)).sum(-1).mean()
        #outputs.loss += kl_loss

        return outputs
