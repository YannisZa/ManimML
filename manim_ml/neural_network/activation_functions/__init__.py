from manim_ml.neural_network.activation_functions.relu import ReLUFunction
from manim_ml.neural_network.activation_functions.sigmoid import SigmoidFunction
from manim_ml.neural_network.activation_functions.linear import LinearFunction
from manim_ml.neural_network.activation_functions.absolute import AbsoluteFunction

name_to_activation_function_map = {
    "ReLU": ReLUFunction, 
    "Sigmoid": SigmoidFunction,
    "Linear": LinearFunction,
    "Absolute": AbsoluteFunction
}


def get_activation_function_by_name(name):
    assert (
        name in name_to_activation_function_map.keys()
    ), f"Unrecognized activation function {name}"

    return name_to_activation_function_map[name]
