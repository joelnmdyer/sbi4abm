from torch import nn

implemented = ["gru", "elman"]

class RNN(nn.Module):

	def __init__(self, input_dim, hidden_dim, num_layers, mlp_dims,
				 flavour="gru", hidden_out_dim=None, N=None, **kwargs):

		super(RNN, self).__init__(**kwargs)

		if not flavour in implemented:
			errmsg = "Kwarg 'flavour' must be in {0}".format(implemented)
			raise ValueError(errmsg)

		if flavour == "elman":	
			self.mod = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
		elif flavour == "gru":
			self.mod = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
		if isinstance(mlp_dims, int):
			output_dim = mlp_dims
			self._layers = []
			self.final = nn.Linear(hidden_dim, output_dim)
		elif isinstance(mlp_dims, list):
			self._layers = [nn.Linear(hidden_dim, mlp_dims[0])]
			for i in range(len(mlp_dims) - 2):
				self._layers.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
			self.final = nn.Linear(mlp_dims[-2], mlp_dims[-1])
		self.relu = nn.ReLU()

	def forward(self, x):

		out, _ = self.mod(x)
		_x = out[:, -1, :]
		for layer in self._layers:
			_x = self.relu(layer(_x))
		return self.final(_x)
