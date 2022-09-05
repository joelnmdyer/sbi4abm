import torch

from torch_geometric_temporal import GConvGRU


class GConvGRUEmbedding(torch.nn.Module):

	def __init__(self, input_dim=-1, hidden_dim=64, hidden_out_dim=16, mlp_dims=[32, 16], N=100, **kwargs):
		super().__init__()

		self._odim = hidden_out_dim
		self._idim = input_dim
		self._N = N

		self.conv = GConvGRU(in_channels=input_dim, out_channels=hidden_out_dim,
							 K=3)
		self.reduce = torch.nn.Linear(hidden_out_dim, 1)
		mlp_layers = [torch.nn.Linear(N, mlp_dims[0])]#, torch.nn.ReLU()]
		for i in range(len(mlp_dims) - 1):
			mlp_layers += [torch.nn.Linear(mlp_dims[i], mlp_dims[i+1])]
			#mlp_layers += [torch.nn.ReLU()]
		self.mlp_layers = torch.nn.Sequential(*mlp_layers)
		print(self.mlp_layers)
		self.relu = torch.nn.ReLU()

	def forward(self, y):

		"""
		y is a (batch_size, T, N, N + feature_dim) tensor, where 

		- T is number of time steps
		- N is number of agents
		- feature_dim is the size of the feature vector associated to each agent. 

		Assumed that the last feature_dim indices are those corresponding to the 
		feature vectors.
		"""

		#y = y.reshape(-1, 51, self._N, y.size(-1))
		batch_size = y.size(0)
		adj_mats, feature_vecs = y[..., :self._N], y[..., self._N:]
		h = None
		for t in range(y.size(1)):
			adj_mat = adj_mats[:, t, ...]
			edge_index = adj_mat.nonzero().T
			edge_index = edge_index[0] * self._N + edge_index[1:]
			edge_weights = adj_mat[adj_mat.nonzero(as_tuple=True)]
			x = feature_vecs[:, t, ...].reshape(-1, feature_vecs.size(-1))
			#print(x.size(), edge_index.size(), edge_weights.size())
			h = self.conv(x, edge_index, edge_weights, H=h)
			#print(h.isnan().sum())
		#h = torch.max(h, dim=1)[0].reshape(batch_size, -1)
		h = self.relu(self.reduce(h).reshape(batch_size, -1))
		return self.mlp_layers(h)
