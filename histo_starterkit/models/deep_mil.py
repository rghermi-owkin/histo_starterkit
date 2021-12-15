import torch
import warnings

from typing import Optional, List

from classic_algos.nn.modules.mlp import MLP
from classic_algos.nn.modules.tile_layers import TilesMLP


class CustomDeepMIL(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = False,
    ):
        super(CustomDeepMIL, self).__init__()

        self.score_model = TilesMLP(
            in_features, hidden=tiles_mlp_hidden, bias=bias, out_features=out_features
        )
        self.score_model.apply(self.weight_initialization)

        mlp_in_features = in_features
        self.mlp = MLP(
            mlp_in_features,
            1,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )
        self.mlp.apply(self.weight_initialization)
        
    @staticmethod
    def weight_initialization(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES, 1), True for values that were padded.
        Returns
        -------
        logits, attention_weights: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, IN_FEATURES, OUT_FEATURES)
        """
        # Attention layer
        w = self.score_model(x=x, mask=mask)
        
        # Normalize weights
        ## Softmax
        #w = torch.nn.functional.softmax(w, dim=1)
        ## Masked softmax
        w = torch.nn.functional.softmax(w.masked_fill(mask, float('-inf')), dim=1)
        
        # Aggregate features
        f = torch.bmm(torch.transpose(x, 1, 2), w)
        f = f.squeeze(2)
        
        return self.mlp(f), w