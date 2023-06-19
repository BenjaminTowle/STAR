import torch

from dataclasses import dataclass
from torch import nn
from typing import Optional

from .matching import DistilBertMatchingModel


@dataclass
class MCVAEOutput:
    x_embeds: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class MCVAEHead(nn.Module):

    def __init__(self, z, model_dim) -> None:
        super().__init__()

        self.q_y = nn.Sequential(
            nn.Linear(model_dim * 2, z),
            nn.Tanh()
        )

        self.log_var = nn.Linear(z, z)
        self.mu = nn.Linear(z, z)

        self.proj = nn.Sequential(
            nn.Linear(model_dim + z, z),
            nn.Tanh(),
            nn.Linear(z, model_dim)
        )

    @staticmethod
    def _get_ground_truth_embeds(
        y_embeds,
        labels=None
    ):
        if y_embeds.ndim == 2:
            return y_embeds

        assert labels is not None, "'labels' must be provided when ground truth 'y_embeds' is ambiguous."
        
        return y_embeds.gather(1, labels.unsqueeze(
                -1).unsqueeze(-1).expand(-1, -1, y_embeds.size(-1))
        ).squeeze(1)

    def forward(
        self,
        x_embeds,
        y_embeds,
        labels=None
    ):

        # Compute posterior
        y_embeds = self._get_ground_truth_embeds(y_embeds, labels)
        q_y = self.q_y(torch.cat([x_embeds, y_embeds], dim=-1))
        mu = self.mu(q_y)
        logvar = self.log_var(q_y)
        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)
        z = eps * std + mu

        y_hat = self.proj(torch.cat([x_embeds, z], dim=-1))
        loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        return MCVAEOutput(
            x_embeds=y_hat,
            logvar=logvar,
            mu=mu,
            loss=loss
        )



class DistilBertMCVAE(DistilBertMatchingModel):
    def __init__(
        self, 
        config, 
        use_symmetric_loss=False, 
        z: int = 256,
        kld_weight: float = 1.0,
        use_kld_annealling: bool = False,
        kld_annealling_steps: int = -1,
        use_message_prior: bool = False
    ):
    
        super().__init__(config, use_symmetric_loss=use_symmetric_loss)

        self.z = z
        self.kld_weight = kld_weight
        self.kld_cur_weight = 0.0 if use_kld_annealling else kld_weight
        self.use_kld_annealling = use_kld_annealling
        self.kld_annealling_steps = kld_annealling_steps
        self.use_message_prior = use_message_prior

        self.cvae = MCVAEHead(z, config.dim)


    def generate_embedding(self, embeds, num_samples=1):
        bsz, dim = embeds.size()
        if bsz == 1:
            embeds = embeds.expand(num_samples, -1)
            z = torch.randn_like(embeds)[:, :self.z]
            embeds = self.cvae.proj(torch.cat([embeds, z], dim=-1))
        else:
            z = torch.randn(bsz * num_samples, self.z).to(embeds.device)
            embeds = embeds.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, dim)
            embeds = self.cvae.proj(torch.cat([embeds, z], dim=-1))
            embeds = embeds.reshape(bsz, num_samples, -1)
        return embeds

    def forward(
            self,
            input_ids: torch.Tensor,
            y_input_ids: torch.Tensor = None,
            candidate_input_ids: torch.Tensor = None,
            candidate_embeds: torch.Tensor = None,
            labels: torch.Tensor = None,
    ):

        with torch.no_grad():
            m_outputs = self.forward_biencoder(
                input_ids=input_ids,
                y_input_ids=y_input_ids,
                candidate_input_ids=candidate_input_ids,
                candidate_embeds=candidate_embeds,
            )

        cvae_outputs = self.cvae(
            x_embeds=m_outputs.x_embeds,
            y_embeds=m_outputs.y_embeds,
            labels=labels
        )

        outputs = self.loss_fn(
            x_embeds=cvae_outputs.x_embeds,
            y_embeds=m_outputs.y_embeds,
            labels=labels
        )
        if self.use_kld_annealling:
            self.kld_cur_weight = min(
                self.kld_cur_weight + 1/self.kld_annealling_steps, self.kld_weight)
        outputs.loss += self.kld_cur_weight * cvae_outputs.loss

        return outputs
