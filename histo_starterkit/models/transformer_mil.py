import torch

from typing import Optional, List, Tuple

from histo_starterkit.models.attention import BasicAttention, GatedAttention
from classic_algos.nn.modules.mlp import MLP
from classic_algos.nn.modules.tile_layers import TilesMLP


class TransformerMIL(torch.nn.Module):

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
        super(TransformerMIL, self).__init__()

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

        # Embedding and positional encoding
        x = self.embedding_layer(x, mask)
        #positions = self.pos_embedding()
        #tiles_emb = tiles_emb + positions

        # Transformer block

        ## Self-attention
        attention_output, _ = self.attention_layer(x)
        x = self.norm_layer(x + attention_result)
        x = self.dropout(x)

        ## Feedforward
        feedforward_output = self.feedforward(x)
        x = self.norm_layer(x + feedforward_output)
        x = self.dropout(x)

        # Classification head
        logits = self.mlp(x)

        return logits, None