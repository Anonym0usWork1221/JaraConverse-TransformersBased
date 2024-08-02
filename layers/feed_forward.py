"""
Resources: https://arxiv.org/abs/2012.14913
"""

from tensorflow import keras
from typing import Any


@keras.saving.register_keras_serializable(package='FeedForward')
class FeedForward(keras.layers.Layer):
    """
    FeedForward class implements a feed-forward neural network with residual connections and layer normalization.

    This class contains a two-layer fully connected neural network with ReLU activation and dropout, followed by
    residual addition and layer normalization. It is used in transformer architectures as the feed-forward component
    in each transformer block.

    Attributes:
        sequential_model (keras.Sequential): A sequential model containing two dense layers and a dropout layer.
        add_layer (keras.layers.Add): Layer that adds the input and the output of the feed-forward network.
        normalization_layer (keras.layers.LayerNormalization): Layer normalization applied after the residual addition.
    """

    def __init__(self, model_dimensions: int, feed_forward_dimensions: int, dropout_rate: float = 0.1):
        """
        Initializes the FeedForward instance.

        Args:
            model_dimensions (int): The number of dimensions in the model.
            feed_forward_dimensions (int): The number of dimensions in the feed-forward network.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.1.
        """

        super().__init__()
        self.sequential_model: keras.Sequential = keras.Sequential([
            keras.layers.Dense(feed_forward_dimensions, activation='relu'),
            keras.layers.Dense(model_dimensions),
            keras.layers.Dropout(dropout_rate)
        ])
        self.add_layer: keras.layers.Add = keras.layers.Add()
        self.normalization_layer: keras.layers.LayerNormalization = keras.layers.LayerNormalization(
            epsilon=1e-7
        )

    def call(self, inputs: Any, *args, **kwargs) -> Any:
        """
        Forward pass for the feed-forward layer.

        Args:
            inputs (Any): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Output tensor after applying the feed-forward network, residual addition, and layer normalization.
        """

        inputs = self.add_layer([inputs, self.sequential_model(inputs)])
        inputs = self.normalization_layer(inputs)
        return inputs


if __name__ == '__main__':
    from tensorflow import expand_dims
    from GlobalVariables import JaraConverseModelConfiguration
    from positional_encoding_and_embeddings import PositionalEmbedding

    embed_code = PositionalEmbedding(
        vocab_size=54000,
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value
    )
    targ_in = expand_dims(input=[1, 536, 10194, 67, 474, 12, 92, 16, 677, 4672], axis=0)
    code_emb = embed_code(targ_in)

    print("===FeedForward===")
    sample_ffn = FeedForward(
        JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value,
        JaraConverseModelConfiguration.FF_DIMENSION.value
    )
    print(code_emb.shape)
    print(sample_ffn(code_emb).shape)

    assert keras.saving.get_registered_object('FeedForward>FeedForward') == FeedForward
    assert keras.saving.get_registered_name(FeedForward) == 'FeedForward>FeedForward'
