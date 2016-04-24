from blocks.bricks import Linear, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from theano import tensor
from blocks_extras.extensions.plot import Plot
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.extensions import FinishAfter, Printing, ProgressBar
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from BrownDataset import BrownDataset
from SaveWeightsExtension import SaveWeights
import pickle
import sys
import os
import logging

logger = logging.getLogger(__name__)

def main():
    if sys.argv:
        epochs = int(sys.argv[1])
        HIDDEN_DIMS = int(sys.argv[2])
        name = "./" + sys.argv[3] + "/"

        if not os.path.exists(name):
            os.makedirs(name)

        run(epochs, HIDDEN_DIMS, name)


def run(epochs=1, HIDDEN_DIMS=100, path="./"):
    brown = BrownDataset()

    INPUT_DIMS = brown.get_vocabulary_size()

    OUTPUT_DIMS = brown.get_vocabulary_size()

    # These are theano variables
    x = tensor.lmatrix('context')
    y = tensor.ivector('output')

    # Construct the graph
    input_to_hidden = LookupTable(name='input_to_hidden', length=INPUT_DIMS,
                                  dim=HIDDEN_DIMS)

    # Compute the weight matrix for every word in the context and then compute
    # the average.
    h = tensor.mean(input_to_hidden.apply(x), axis=1)

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

    W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
    cost = cost + 0.01 * (W1 ** 2).sum() + 0.01 * (W2 ** 2).sum()
    cost.name = 'cost_with_regularization'

    mini_batch = SequentialScheme(brown.num_instances(), 250)
    data_stream = DataStream.default_stream(brown, iteration_scheme=mini_batch)

    # Now we tie up lose ends and construct the algorithm for the training
    # and define what happens in the main loop.
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                                step_rule=Scale(learning_rate=0.1))

    extensions = [
        ProgressBar(),
        FinishAfter(after_n_epochs=epochs),
        Printing(),
        # TrainingDataMonitoring(variables=[cost]),
        SaveWeights(layers=[input_to_hidden, hidden_to_output],
                    prefixes=['%sfirst' % path, '%ssecond' % path]),
        # Plot(
        #     'Word Embeddings',
        #     channels=[
        #         [
        #             'cost_with_regularization'
        #         ]
        #     ])
    ]

    logger.info("Starting main loop...")
    main = MainLoop(data_stream=data_stream,
                    algorithm=algorithm,
                    extensions=extensions)

    main.run()

    pickle.dump(cg, open('%scg.pickle' % path, 'wb'))

if __name__ == '__main__':
    main()
