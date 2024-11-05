from manim import *

from manim_ml.neural_network.layers.parent_layers import VGroupNeuralNetworkLayer
import manim_ml

class OutputLayer(VGroupNeuralNetworkLayer):
    """Handles rendering a layer for a neural network"""

    def __init__(
        self,
        node_text:VGroup,
        layer_buffer=SMALL_BUFF / 2,
        node_radius=0.08,
        node_color=manim_ml.config.color_scheme.secondary_color,
        text_color=manim_ml.config.color_scheme.text_color,
        node_outline_color=manim_ml.config.color_scheme.secondary_color,
        rectangle_color=manim_ml.config.color_scheme.secondary_color,
        node_spacing=0.3,
        rectangle_fill_color=manim_ml.config.color_scheme.background_color,
        node_stroke_width=2.0,
        rectangle_stroke_width=2.0,
        animation_dot_color=manim_ml.config.color_scheme.active_color,
        activation_function=None,
        **kwargs
    ):
        super(VGroupNeuralNetworkLayer, self).__init__(**kwargs)
        self.node_text = node_text
        self.num_nodes = len(node_text)
        self.layer_buffer = layer_buffer
        self.node_radius = node_radius
        self.node_color = node_color
        self.text_color = text_color
        self.node_stroke_width = node_stroke_width
        self.node_outline_color = node_outline_color
        self.rectangle_stroke_width = rectangle_stroke_width
        self.rectangle_color = rectangle_color
        self.node_spacing = node_spacing
        self.rectangle_fill_color = rectangle_fill_color
        self.animation_dot_color = animation_dot_color
        self.activation_function = activation_function

        self.node_group = VGroup()
        self.text_group = VGroup()

    def construct_layer(
        self,
        input_layer: "NeuralNetworkLayer",
        output_layer: "NeuralNetworkLayer",
        **kwargs
    ):
        """Creates the neural network layer"""
        # Add Nodes
        for node_number in range(self.num_nodes):
            node_shape = Circle(
                radius=self.node_radius,
                color=self.node_color,
                stroke_width=self.node_stroke_width,
            )
            self.node_text[node_number].move_to(node_shape.get_center())
            node_object = VGroup(node_shape,self.node_text[node_number])

            self.node_group.add(node_object)
            self.text_group.add(self.node_text[node_number])
        # Space the nodes
        # Assumes Vertical orientation
        for node_index, node_object in enumerate(self.node_group):
            location = node_index * self.node_spacing
            node_object.move_to([0, location, 0])
        # Create Surrounding Rectangle
        self.surrounding_rectangle = SurroundingRectangle(
            self.node_group,
            color=self.rectangle_color,
            fill_color=self.rectangle_fill_color,
            fill_opacity=1.0,
            buff=self.layer_buffer,
            stroke_width=self.rectangle_stroke_width,
        )
        self.surrounding_rectangle.set_z_index(1)
        # Add the objects to the class
        self.add(self.surrounding_rectangle, self.node_group)

        super().construct_layer(input_layer, output_layer, **kwargs)


    def make_forward_pass_animation(self, layer_args={}, **kwargs):
        # Make highlight animation
        succession = Succession(
            ApplyMethod(
                self.node_group.set_color, self.animation_dot_color, run_time=0.2
            ),
            Wait(0.5),
            AnimationGroup(
                ApplyMethod(self.node_group.set_color, self.node_color, run_time=0.2),
                ApplyMethod(self.text_group.set_color, self.text_color, run_time=0.2),
            )
        )
        return succession
    
    def make_backward_pass_animation(self, layer_args={}, **kwargs):
        return self.make_forward_pass_animation(
            layer_args=layer_args,
            **kwargs
        )

    @override_animation(Create)
    def _create_override(self, **kwargs):
        animations = []

        animations.append(Create(self.surrounding_rectangle))

        for node in self.node_group:
            animations.append(Create(node))

        animation_group = AnimationGroup(*animations, lag_ratio=0.0)
        return animation_group

    def get_height(self):
        return self.surrounding_rectangle.height

    def get_center(self):
        return self.surrounding_rectangle.get_center()

    def get_left(self):
        return self.surrounding_rectangle.get_left()

    def get_right(self):
        return self.surrounding_rectangle.get_right()

    def move_to(self, mobject_or_point):
        """Moves the center of the layer to the given mobject or point"""
        layer_center = self.surrounding_rectangle.get_center()
        if isinstance(mobject_or_point, Mobject):
            target_center = mobject_or_point.get_center() 
        else:
            target_center = mobject_or_point

        self.shift(target_center - layer_center)