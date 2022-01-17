import torch

from typing import Optional, List, Tuple

from histo_starterkit.models.attention import BasicAttention, GatedAttention
from classic_algos.nn.modules.mlp import MLP
from classic_algos.nn.modules.tile_layers import TilesMLP


class DeepMIL(torch.nn.Module):
    """
    Deep MIL classification model.
    https://arxiv.org/abs/1802.04712
    Example:
        >>> module = DeepMIL(in_features=128, out_features=1)
        >>> logits, attention_scores = module(slide, mask=mask)
        >>> attention_scores = module.score_model(slide, mask=mask)
    Parameters
    ----------
    in_features: int
    out_features: int = 1
    d_model_attention: int = 128
    temperature: float = 1.0
        GatedAttention softmax temperature.
    tiles_mlp_hidden: Optional[List[int]] = None
    mlp_hidden: Optional[List[int]] = None
    mlp_dropout: Optional[List[float]] = None,
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        d_model_attention: int = 128,
        temperature: float = 1.0,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        super(CustomDeepMIL, self).__init__()

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        self.dropout = torch.nn.Dropout(0.20)

        self.tiles_emb = TilesMLP(
            in_features,
            hidden=tiles_mlp_hidden,
            bias=bias,
            out_features=d_model_attention,
        )

        self.attention_layer = BasicAttention(
            d_model=d_model_attention, 
            temperature=temperature,
        )

        mlp_in_features = d_model_attention

        self.mlp = MLP(
            in_features=mlp_in_features,
            out_features=out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )

        dropout = torch.nn.Dropout(0.5)

        self.h = torch.randn((d_model_attention,), device='cuda:1', requires_grad=True)

    @classmethod
    def classification(cls, in_features: int, out_features: Optional[int] = 1) -> "DeepMIL":
        """
        DeepMIL operator with good defaults parameters for classification.
        Parameters
        ----------
        in_features: int
        out_features: Optional[int] = 1
        Returns
        -------
        deep_mil: DeepMIL
        """
        return cls(
            in_features=in_features,
            out_features=out_features,
            d_model_attention=128,
            temperature=1.0,
            tiles_mlp_hidden=None,
            mlp_hidden=[128, 64],
            mlp_dropout=None,
            mlp_activation=torch.nn.ReLU(),
            bias=True,
        )

    @classmethod
    def survival(cls, in_features: int, out_features: Optional[int] = 1) -> "DeepMIL":
        """
        DeepMIL operator with good defaults parameters for survival.
        Parameters
        ----------
        in_features: int
        out_features: Optional[int] = 1
        Returns
        -------
        deep_mil: DeepMIL
        """
        return cls(
            in_features=in_features,
            out_features=out_features,
            d_model_attention=128,
            temperature=1.0,
            tiles_mlp_hidden=None,
            mlp_hidden=[128],
            mlp_dropout=None,
            mlp_activation=None,
            bias=True,
        )

    def score_model(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Gets attention logits.
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.
        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        tiles_emb = self.tiles_emb(x, mask)
        attention_logits = self.attention_layer.attention(tiles_emb, mask)
        return attention_logits

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.
        Returns
        -------
        logits, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, OUT_FEATURES), (B, N_TILES)
        """
        
        tiles_emb = self.tiles_emb(x, mask)
        scaled_tiles_emb, attention_weights = self.attention_layer(tiles_emb, mask)
        scaled_tiles_emb = self.h + scaled_tiles_emb
        scaled_tiles_emb = self.dropout(scaled_tiles_emb)
        logits = self.mlp(scaled_tiles_emb)
        return logits, attention_weights

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
        #f = torch.bmm(w.transpose(1, 2), x)
        f = torch.matmul(w.transpose(1, 2), x)
        f = f.squeeze(1)
        
        return self.mlp(f), w