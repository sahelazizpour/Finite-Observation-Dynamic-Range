# First step of approximation: fit the data to a beta distribution
##################
import numpy as np
from scipy import stats, signal, optimize


def fit_beta_distribution(data, delta, seed=1234):
    for i in range(10):
        try:
            np.random.seed(seed)
            # add very small jitter to avoid problems with delta distributions
            data = data + np.random.randn(len(data)) * delta / 7
            # IMPORTANT: The beta distribution is defined on the interval [0,1] but because of the discretization we need to shift the support by delta and scale it by 1+2*delta
            a, b, loc, scale = stats.beta.fit(data, floc=-delta, fscale=1 + 2 * delta)
            return a, b, loc, scale
        except:
            seed += 1


# Second step of approximation: fit the beta parameters with a neural network
##################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class DenseNN(nn.Module):
    """
    This class implements a fully connected neural network with a variable number of hidden layers.
    Example
    -------
    >> model = DenseNN(input_size=3, hidden_sizes=[42,42,42], output_size=2)
    >> model(torch.tensor([1,2,3], dtype=torch.float32))
    Parameters
    ----------
    input_size: int
        Number of input neurons.
    hidden_sizes: list
        List of hidden layer sizes.
    output_size: int
        Number of output neurons.
    act: torch.nn.Module
        Activation function (default nn.Tanh()).
    """
    def __init__(self, input_size, hidden_sizes, output_size, act=nn.Tanh()):
        super().__init__()
        # create activation function
        self.act = act
        # create layers
        self.hidden = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(1, len(hidden_sizes)):
            self.hidden.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
        self.output = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer in self.hidden:
            x = self.act(layer(x))
            
        return self.output(x)


class FunctionApproximation:
    """
    This class implements a pipeline to preprocess input, to feed it to a neural network (general function approximator), and to postprocess the output.
    Example
    -------
    initialize function approximation
    >> func = FunctionApproximation(input=['lambda', 'window', 'h'], output=['a', 'b'])
    specify a map from input to NN (inverse not needed): since parameters are logarithmically distributed we map them to log scale (for lambda we use 1-lambda because we are interested in getting close to 1)
    >> func.set_map('lambda', lambda x: np.log10(1-x))
    >> func.set_map('window', lambda x: np.log10(x))
    >> func.set_map('h',      lambda x: np.log10(x))
    specify a map from NN to output and its inverse (for training!)
    >> func.set_map_output('a', lambda Y: 10**Y, lambda y: np.log10(y))
    >> func.set_map_output('b', lambda Y: 10**Y, lambda y: np.log10(y))
    training (takes care of scaling and shuffling)
    >> loss = func.train(dataframe, epochs=10000, batch_size=300)
    usage
    >> func([0.9,1,1e-3]) # returns a,b

    Parameters
    ----------
    input: list
        List of input variables.
    output: list
        List of output variables.
    """

    def __init__(self, input_names, output_names, model=None, verbose=False):
        # assert that input and output lists do not share any elements
        assert len(set(input_names).intersection(set(output_names))) == 0
        self.input_names = input_names
        self.output_names = output_names
        self.input_dim = len(self.input_names)
        self.output_dim = len(self.output_names)
        self.map_input = [lambda x: x for i in range(self.input_dim)]
        self.map_output = [lambda x: x for i in range(self.output_dim)]
        self.map_output_inv = [lambda x: x for i in range(self.output_dim)]
        self.verbose = verbose  
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        if self.verbose: 
            print("Using device: {}".format(self.device))

        if model==None:
            self.model = DenseNN(input_size=self.input_dim, hidden_sizes=[42, 42, 42], output_size=self.output_dim)

        if verbose:
            print(self.model)
        
        # Attention: scaler has to match the activation function:
        # tanh: [-1,1]
        # sigmoid: [0,1]    
        self.X_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.Y_scaler = MinMaxScaler(feature_range=(-1, 1))
        # for now: throw error if self.model.act is not nn.Tanh()        

    def change_map(self, label, func, func_inv=None):
        """
            Specifies a mapping for parameter `name` from either input or output to the NN.
            The pipeline is thus as follows: input -> func -> NN -> func_inv -> output
        """
        # find index of name in input_names
        if label in self.input_names:
            i = self.input_names.index(label)
            self.map_input[i] = func
        elif label in self.output_names:
            i = self.output_names.index(label)
            self.map_output[i] = func
            if func_inv is None:
                raise ValueError("Please specify an inverse mapping for the output!")
            self.map_output_inv[i] = func_inv
    
    def __call__(self, x):
        """
            Returns the function approximation at `x`.
        """
        if self.model.training:
            raise ValueError("FunctionApproximation has not finished training yet!")
        X = x.copy()
        for i in range(self.input_dim):
            X[:, i] = self.map_input[i](x[:, i])
        
        X_scaled = self.X_scaler.transform(X)
        Y_scaled = self.model(torch.from_numpy(X_scaled).float()).detach().numpy()
        Y = self.Y_scaler.inverse_transform(Y_scaled)
        y = Y.copy()
        for i in range(self.output_dim):
            y[:, i] = self.map_output_inv[i](Y[:, i])
        return y

    def train(self, dataframe, epochs=1000, lr=0.001):
        """
        Trains the neural network.
        """
        # retrieve training data from dataframe into numpy arrays
        Xs = dataframe[self.input_names].values
        Ys = dataframe[self.output_names].values

        # map inputs and outputs so that data is equidistant (e.g. into logspace if data comes from a log-normal distribution)
        for i in range(self.input_dim):
            Xs[:, i] = self.map_input[i](Xs[:, i])
        for i in range(self.output_dim):
            Ys[:, i] = self.map_output[i](Ys[:, i])

        # rescale data into [-1,1] range    
        Xs = self.X_scaler.fit_transform(Xs)
        Ys = self.Y_scaler.fit_transform(Ys)

        # convert to torch tensors
        X_tensor = torch.from_numpy(Xs).float()
        Y_tensor = torch.from_numpy(Ys).float()
        if self.verbose:
            print(f"Training data with input shape {X_tensor.shape} and output shape {Y_tensor.shape}.")

        # Adam and MSE Loss
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss(reduction='mean')

        # TODO in each epoch make a random sample of the data?
        # TODO: check why this is slower than the reference implementation that should be identical but also result is different ...
        history_loss = []
        for epoch in tqdm(range(epochs), desc="Training"):
            #permutation = torch.randperm(len(self.data))
            Y_pred = self.model(X_tensor)
            loss = loss_fn(Y_pred, Y_tensor)
            history_loss.append(loss.item())
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
           
        self.model.train(False)
        if self.verbose:
            print("Training complete; left training mode for faster evaluation! To re-enter training mode call `func.model.train(True)`.")

        return history_loss
            

 
