from blocks.bricks import Linear, Softmax
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Scale
from blocks.main_loop import MainLoop
from theano import tensor
from blocks.extensions import FinishAfter, Printing
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from BrownDataset import BrownDataset

INPUT_DIMS = 500
HIDDEN_DIMS = 100
OUTPUT_DIMS = 200

context = 1

brown = BrownDataset()

# context_matrix = TensorType('float32', (False,)*context)
# x = context_matrix('features')

# These are theano variables
x = tensor.lmatrix('context')
y = tensor.fmatrix('output')

# Construct the graph
input_to_hidden = Linear(name='input_to_hidden', input_dim=INPUT_DIMS,
                         output_dim=HIDDEN_DIMS)

# Compute the weight matrix for every word in the context and then compute
# the average.
# TODO Test if one could simply compute an average input vector beforehand
h = tensor.mean(input_to_hidden.apply(x))

print(h)

hidden_to_output = Linear(name='hidden_to_output', input_dim=HIDDEN_DIMS,
                          output_dim=OUTPUT_DIMS)
y_hat = Softmax().apply(hidden_to_output.apply(h))


# And initialize with random varibales and set the bias vector to 0
weights = IsotropicGaussian(0.01)
input_to_hidden.weights_init = hidden_to_output.weights_init = weights
input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)
input_to_hidden.initialize()
hidden_to_output.initialize()

# And now the cost function
cost = CategoricalCrossEntropy().apply(y, y_hat)
cg = ComputationGraph(cost)

data_stream = DataStream.default_stream(brown,
                iteration_scheme=SequentialScheme(brown.num_instances(), 10))

# Now we tie up lose ends and construct the algorithm for the training
# and define what happens in the main loop.
algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                            step_rule=Scale(learning_rate=0.1))

extensions = [
    FinishAfter(after_n_epochs=1),
    Printing()
]

main = MainLoop(data_stream=data_stream,
                algorithm=algorithm,
                extensions=extensions)

main.run()
