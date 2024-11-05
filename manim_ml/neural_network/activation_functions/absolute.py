from manim import *
import numpy as np

import manim_ml
from manim_ml.neural_network.activation_functions.activation_function import (
    ActivationFunction,
)

class AbsoluteFunction(ActivationFunction):
    """Absolute Activation Function"""

    def __init__(
            self, 
            function_name="Absolute", 
            x_range=[-1, 1], 
            y_range=[-1, 1]
        ):
        super().__init__(function_name, x_range, y_range)

    def apply_function(self, x_val):
        return np.abs(x_val)