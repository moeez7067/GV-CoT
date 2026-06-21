import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ReasoningGAT(nn.Module):
    """2-layer Graph Attention Network for node-level classification.

    Each node is a context sentence embedded with SBERT (384-dim).
    The model classifies each node as valid/gold (1) or hallucinated/distractor (0).

    Args:
        in_channels:     Dimensionality of input node features (default: 384).
        hidden_channels: Width of the first GAT layer's output (default: 128).
        out_channels:    Number of output classes (default: 2).
        heads:           Number of attention heads in the first layer (default: 4).
        dropout:         Dropout probability applied to node features (default: 0.3).
    """

    def __init__(
        self,
        in_channels: int = 384,
        hidden_channels: int = 128,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: multi-head attention; output is heads * hidden_channels
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        # Layer 2: single-head attention collapses back to hidden_channels
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:          Node feature matrix [num_nodes, in_channels].
            edge_index: Graph connectivity [2, num_edges].

        Returns:
            Log-softmax class scores per node [num_nodes, out_channels].
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))

        return F.log_softmax(self.classifier(x), dim=-1)
