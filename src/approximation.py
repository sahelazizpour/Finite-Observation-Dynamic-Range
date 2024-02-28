# First step of approximation: fit the data to a beta distribution
##################
import numpy as np
from scipy import stats, signal, optimize
import os
import pickle
import inspect
import h5py


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
        self.input_size = input_size
        self.output_size = output_size
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

    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose
        if "params" in kwargs:
            self.params = kwargs["params"]
        else:
            self.params = None
        if "filename" in kwargs:
            filename = kwargs["filename"]
            if self.verbose:
                print(
                    f"Loading function approximation from {filename} and ignore additional arguments!"
                )
            self.load(filename)
        else:
            self.init(**kwargs)

    def init(self, **kwargs):
        self.input_names = kwargs["input"]
        self.output_names = kwargs["output"]
        assert len(set(self.input_names).intersection(set(self.output_names))) == 0
        self.input_dim = len(self.input_names)
        self.output_dim = len(self.output_names)
        self.map_input = [lambda x: x for i in range(self.input_dim)]
        self.map_output = [lambda x: x for i in range(self.output_dim)]
        self.map_output_inv = [lambda x: x for i in range(self.output_dim)]
        # ranges for input that the model is trained on (initialize with None)
        self.input_range = [(None, None) for i in range(self.input_dim)]

        if "model" in kwargs:
            self.model = kwargs["model"]
            assert self.model.input_size == self.input_dim
            assert self.model.output_size == self.output_dim
        else:
            if self.verbose:
                print(
                    "Initializing default model (choose own model with kwargs `model`)."
                )
            self.model = DenseNN(
                input_size=self.input_dim,
                hidden_sizes=[42, 42, 42],
                output_size=self.output_dim,
            )
        if self.verbose:
            print(self.model)

        # Attention: scaler has to match the activation function:
        # tanh: [-1,1]
        # sigmoid: [0,1]
        self.X_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.Y_scaler = MinMaxScaler(feature_range=(-1, 1))

        if self.verbose:
            print(
                "Next steps:\n>> func.set_map() # optimize parameter distribution over activation function domain (input) and range (output) \n>> func.train() # train neural network model\n>> func.save(filename) # save function approximation to file"
            )

    def set_map_to_NN(self, label, func):
        """
        Specifies a mapping for parameter `name` from either input or output to the NN.
        The pipeline is thus as follows: input -> func -> NN -> func_inv -> output
        """
        if label in self.input_names:
            i = self.input_names.index(label)
            self.map_input[i] = func
        elif label in self.output_names:
            i = self.output_names.index(label)
            self.map_output[i] = func

    def set_map_from_NN(self, label, func):
        """
        Specifies a mapping for parameter `name` from either input or output to the NN.
        The pipeline is thus as follows: input -> func -> NN -> func_inv -> output
        """
        if label in self.output_names:
            i = self.output_names.index(label)
            self.map_output_inv[i] = func
        elif label in self.input_names:
            raise ValueError("Inverse mapping only required for output variables!")
        
    def prepare_data(self, dataframe):
        assert self.model.training
        
        # extract training data from dataframe into numpy arrays
        Xs = dataframe[self.input_names].values
        Ys = dataframe[self.output_names].values

        # specify input ranges used for training
        for i in range(self.input_dim):
            self.input_range[i] = (np.min(Xs[:, i]), np.max(Xs[:, i]))

        # map inputs and outputs so that data is equidistant
        # (e.g. into logspace if data comes from a log-normal distribution)
        for i in range(self.input_dim):
            Xs[:, i] = self.map_input[i](Xs[:, i])
        for i in range(self.output_dim):
            Ys[:, i] = self.map_output[i](Ys[:, i])

        # rescale data into [-1,1] range (this restricts the fit to the range of the training data!!)
        Xs = self.X_scaler.fit_transform(Xs)
        Ys = self.Y_scaler.fit_transform(Ys)

        return Xs, Ys

    def train(self, Xs, Ys, custom_loss=None, epochs=1000, lr=0.005, device=None):
        """
        Trains the neural network.
        """
        # select suitable device for training
        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if self.verbose:
            print("Device for training: {}".format(device))

        # convert to torch tensors
        X_tensor = torch.from_numpy(Xs).float().to(device)
        Y_tensor = torch.from_numpy(Ys).float().to(device)
        model_ = self.model.to(device)
        if self.verbose:
            print(
                f"Training data with input shape {X_tensor.shape} and output shape {Y_tensor.shape}."
            )

        # Adam and MSE Loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss(reduction="mean")
        if custom_loss is None:
            custom_loss = lambda Y_pred, Y, X: loss_fn(Y_pred, Y)

        history_loss = []
        for epoch in tqdm(range(epochs), desc="Training"):
            # forward pass
            Y_pred = model_(X_tensor)
            loss = custom_loss(Y_pred, Y_tensor, X_tensor)
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            # for logging
            history_loss.append(loss.item())
        # fetch model from device
        self.model = model_.to("cpu")
        # get data from device (does this free memory on the device?)
        Xs = X_tensor.to("cpu").detach().numpy()
        Ys = Y_tensor.to("cpu").detach().numpy()
        # set training to false to enter evaluation mode
        self.model.train(False)
        if self.verbose:
            print(
                "Training complete; left training mode for faster evaluation! To re-enter training mode call `func.model.train(True)`."
            )

        return history_loss

    def save(self, filename):
        """
        Saves the function approximation to a pickle file.
        """

        # step-by-step save all the relevant objects for running the function approximation (not training!)
        def to_string(lambda_function):
            # getsource returns "func.set_map_to_NN('lambda', lambda x: f(x) )\n" or "func.set_map_from_NN('lambda', lambda x: f_inv(x) )"
            # we want to return "lambda x: f(x)"
            return inspect.getsource(lambda_function).split(",")[1][:-2]

        # create datastructure that can be iteratively passed to pickle.dump
        data = {}
        data["params"] = self.params
        data["input_names"] = self.input_names
        data["output_names"] = self.output_names
        # convert lambda functions to strings
        data["map_input"] = [to_string(m) for m in self.map_input]
        data["map_output"] = [to_string(m) for m in self.map_output]
        data["map_output_inv"] = [to_string(m) for m in self.map_output_inv]
        data["X_scaler"] = self.X_scaler
        data["Y_scaler"] = self.Y_scaler
        data["model"] = self.model
        data["input_range"] = self.input_range

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename):
        """
        Loads class objects from a pickle file.
        """
        data = pickle.load(open(filename, "rb"))
        self.params = data["params"]
        self.input_names = data["input_names"]
        self.output_names = data["output_names"]
        self.input_dim = len(self.input_names)
        self.output_dim = len(self.output_names)
        # convert strings back to lambda functions
        self.map_input = [eval(m) for m in data["map_input"]]
        self.map_output = [eval(m) for m in data["map_output"]]
        self.map_output_inv = [eval(m) for m in data["map_output_inv"]]
        self.X_scaler = data["X_scaler"]
        self.Y_scaler = data["Y_scaler"]
        self.model = data["model"]
        self.input_range = data["input_range"]
        self.model.train(False)

    def __call__(self, *x):
        """
        Returns the approximation of y1,y2, ... = func(x1,x2,...) where x1,x2 are the arguments in the same order as specified in `func.input_names` and y1, y2, ... are the outputs in the same order as specified in `func.output_names`.

        xi currently have to be lists or numpy arrays of the same length.
        """
        if self.model.training:
            raise ValueError("FunctionApproximation has not finished training yet!")

        x = list(x)

        if len(x) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input values, but got {len(x)}!"
            )

        # iterate over inputs to transform scalars into arrays and apply mapping
        for i in range(self.input_dim):
            if np.isscalar(x[i]):
                x[i] = np.array([x[i]])
            else:
                x[i] = np.array(x[i])

            # apply mapping
            x[i] = self.map_input[i](x[i])

        # stack input list into one array
        X = np.stack(x, axis=1)

        X_scaled = self.X_scaler.transform(X)
        Y_scaled = self.model(torch.from_numpy(X_scaled).float()).detach().numpy()
        Y = self.Y_scaler.inverse_transform(Y_scaled)

        # unstack output array into separate arrays
        y = [Y[:, i] for i in range(self.output_dim)]
        # apply inverse mapping
        for i in range(self.output_dim):
            y[i] = self.map_output_inv[i](y[i])
            # if output is a scalar, return scalar
            if len(y[i]) == 1:
                y[i] = y[i][0]

        return tuple(y)


# Third step of approximation: Use approximation in analysis
##################
import sqlite3
from src.utils import *
from src.analysis import *
from src.theory import *
from dask.distributed import Client, LocalCluster, as_completed


def analysis_beta_approximation(
    params, list_lambda, database, verbose=False, redo=False
):
    """
    Uses beta_interpolation based on simulation results from `database` to estimate the number of discriminable intervals and dynamic range for the given parameters `params`
    Stores result in `path_out`
    Function parallelizes by splitting up different lamda values into different processes

    Parameters
    ----------
    params : dict
        Dictionary with simulation parameters.
    database : str
        Path to database.
    """
    # get relevant information from database
    import sqlite3

    con = sqlite3.connect(database)
    cur = con.cursor()

    if exists_in_database(con, cur, "results", params) & (not redo):
        # warning if results already exist and return what is in database
        if verbose:
            print(f"Results already in database for params; load from database.")
        return load_analysis_beta_approximation(params, database=database)

    # load function approximation from database
    beta_interpolation = pd.read_sql_query(
        f"SELECT * FROM beta_interpolations WHERE N={params['N']} AND K={params['K']} AND mu={params['mu']} AND seed={params['seed']}",
        con,
    )
    # check that there is a unique function approximation
    if not len(beta_interpolation) == 1:
        raise ValueError(
            f"No unique function approximation in database for params (either 0 or multiple entries)"
        )
    if verbose:
        print(
            f"Load beta interpolation from file: {beta_interpolation['filename'].values[0]}"
        )
    beta_approx = FunctionApproximation(
        filename=beta_interpolation["filename"].values[0]
    )
    con.close()

    # define pmf from convolution of beta distribution with Gaussian noise
    delta = 1 / beta_approx.params["N"]
    support = np.arange(0, 1 + 4 * params["sigma"], delta)
    support = np.concatenate((-support[::-1], support[1:]))
    loc = beta_approx.params["loc"]
    scale = beta_approx.params["scale"]

    def ml_pmf(window, lam, h, verbose=False):
        a, b = beta_approx(lam, window, h)
        # pmf as difference of cdf to ensure that the pmf is normalized
        pmf_beta = np.diff(stats.beta.cdf(support, a, b, loc=loc, scale=scale))
        # convolution with a Gaussian distribution at every point of the support
        pmf_norm = stats.norm.pdf(support, 0, params["sigma"]) * delta
        return np.convolve(pmf_beta, pmf_norm, mode="same")

    # parallel processing of different lambda values and store results in dataframe
    df = pd.DataFrame(columns=["lambda", "number_discriminable", "dynamic_range"])

    # specify h_range (math the range function is trained on)
    h_range = beta_approx.input_range[beta_approx.input_names.index("h")]
    if verbose:
        print(f"Using h_range = {h_range}")

    # define function for dask
    def analyse(lam):
        """
        return lambda, number of discriminable intervals, dynamic range
        """

        def pmf_o_given_h(h):
            return ml_pmf(params["window"], lam, h)

        # activity_left = mean_field_activity(lam, params["mu"], 0)
        # pmf_ref_left =  stats.norm.pdf(support, activity_left, params["sigma"]) * delta
        # activity_right = mean_field_activity(lam, params["mu"], 1e3)
        # pmf_ref_right = stats.norm.pdf(support, activity_right, params["sigma"]) * delta
        pmf_ref_left = pmf_o_given_h(h_range[0])
        pmf_ref_right = pmf_o_given_h(h_range[-1])
        pmf_refs = [pmf_ref_left, pmf_ref_right]

        hs_left = find_discriminable_inputs(
            pmf_o_given_h, h_range, pmf_refs, params["epsilon"]
        )
        hs_right = find_discriminable_inputs(
            pmf_o_given_h,
            h_range,
            pmf_refs,
            params["epsilon"],
            start="right",
        )
        if len(hs_left) > 0 and len(hs_right) > 0:
            return (
                lam,
                0.5 * (len(hs_left) + len(hs_right)),
                dynamic_range((hs_left[0], hs_right[0])),
            )
        else:
            return lam, np.nan, np.nan

    # execute independent lambda computations in parallel with dask
    cluster = LocalCluster()
    dask_client = Client(cluster)

    futures = dask_client.map(analyse, list_lambda)

    # run analysis
    data = []
    for future in tqdm(as_completed(futures), total=len(list_lambda)):
        data.append(future.result())

    # sort data by first column
    data = np.array(sorted(data, key=lambda x: x[0]))

    # return dictionary of results
    return {
        "params": params,
        "data": data,
    }


def save_analysis_beta_approximation(
    result, path="./dat/", database="./simulations.db"
):
    params = result["params"]
    # write results to file and database
    # save dataframe to ASCI files (tab-separated)
    filename = f"{path}/sigma={params['sigma']}_epsilon={params['epsilon']}/N={params['N']}_K={params['K']}_mu={params['mu']}/results_simulation_seed={params['seed']}_window={params['window']}.txt"
    # create directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # save data to file
    np.savetxt(
        filename,
        result["data"],
        delimiter="\t",
        header="#lambda\tnumber of discriminable inputs\tdynamic_range",
        comments="",
    )
    params["filename"] = filename

    # store in database
    con = sqlite3.connect(database)
    cur = con.cursor()
    insert_into_database(con, cur, "results", params)
    con.commit()
    con.close()

def load_analysis_beta_approximation(
    params, path="./dat/", database="./simulations.db"    
):
    con = sqlite3.connect(database)
    cur = con.cursor()
    filename = fetch_from_database(con, cur, "results", params)[-1][-1]
    con.close()
    print(filename)
    data = np.loadtxt(filename, delimiter="\t", skiprows=1)
    
    return {
        "params": params,
        "data": data,
    }
