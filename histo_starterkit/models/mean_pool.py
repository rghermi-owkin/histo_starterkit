import torch

from typing import Optional, List

from classic_algos.nn import MLP


class MeanPool(torch.nn.Module):
    """
    MeanPool Module
    Parameters
    ----------
    in_features: int
    out_features: int
    hidden: Optional[List[int]] = None
    dropout: Optional[List[float]] = None,
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        super(MeanPool, self).__init__()
        
        self.mlp = MLP(
            in_features=in_features,
            out_features=out_features,
            hidden=hidden,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.mlp.apply(self.weight_initialization)
    
    @staticmethod
    def weight_initialization(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.
        Returns
        -------
        logits: torch.Tensor
            (B, OUT_FEATURES)
        """
        if mask is not None:
            # Only mean over non padded features
            mean_x = torch.sum(x.masked_fill(mask, 0.0), dim=1) / torch.sum(
                (~mask).float(), dim=1
            )
        else:
            mean_x = torch.mean(x, dim=1)

        return self.mlp(mean_x), None