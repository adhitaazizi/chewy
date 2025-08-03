# Copyright (c) 2025, Chunan Liu and Aurélien Plüsser
# All rights reserved.
#
# This source code is licensed under the CC BY-NC-SA 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch_geometric.data import Data as PygData


# create a PairData object that inherits PyGData object,
# attrs:
# Ab and Ag node features      => x_b and x_g
# Ab and Ag inner graph edges  => edge_index_b and edge_index_g
# Ab-Ag bipartite edges        => edge_index_bg
class PairData(PygData):
    # define how to increment the edge_index_b and edge_index_g
    # when concatenating multiple PairData objects
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index_b":
            # return the number of the Ab nodes
            return self.x_b.size(0)
        if key == "edge_index_g":
            # return the number of the Ag nodes
            return self.x_g.size(0)
        if key == "edge_index_bg":
            # return the number of the Ab and Ag nodes
            return torch.tensor([[self.x_b.size(0)], [self.x_g.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)