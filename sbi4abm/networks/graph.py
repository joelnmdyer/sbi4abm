import torch

from torch_geometric.nn import GCNConv


class GCNEncoder(torch.nn.Module):
	def __init__(self, input_dim=-1, hidden_dim=64, hidden_out_dim=16, mlp_dims=[32, 16], N=100, **kwargs):
		super().__init__()

		self._odim = hidden_out_dim
		self._idim = input_dim

		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, hidden_out_dim)
		self.mlp_layers = [torch.nn.Linear(N, mlp_dims[0])]
		self.mlp_layers += [torch.nn.Linear(mlp_dims[i], mlp_dims[i+1])
						   for i in range(len(mlp_dims) - 1)]
		#if input_dim == -1:
		#	self.x = None
		#else:
		#	self.x = torch.ones(input_dim).unsqueeze(-1)

	def forward(self, y):

		"""
		y is a (weighted) adjacency matrix - assume no node features
		"""

		# Extract from y an edge list and edge weights
		batch_size = y.size(0)
		N = y.size(1)
		#y = torch.block_diag(*y)
		edge_index = y.nonzero().T
		edge_index = edge_index[0] * N + edge_index[1:]
		edge_weights = y[y.nonzero(as_tuple=True)]
		x = torch.ones(batch_size * N).unsqueeze(-1)
		x = self.conv1(x, edge_index, edge_weights)
		x = x.relu()
		x = self.conv2(x, edge_index, edge_weights)
		# Max pooling across node embedding elements
		x = torch.max(x, dim=1)[0]
		x = x.reshape(batch_size, -1)
		for layer in self.mlp_layers:
			x = layer(x)
		return x
