"""
This module defines positional encoding functions and a PositionalEmbedding layer,
which are essential for Transformer-based models.

Resources: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
"""

from numpy import arange, newaxis, concatenate, sin, cos, ndarray, dtype, floating, timedelta64, bool_
from GlobalVariables import JaraConverseModelConfiguration
from tensorflow import keras, cast, float32, shape, math
from typing import Union, Any, Optional


@keras.saving.register_keras_serializable(package='positional_encoding')
def positional_encoding(length: int, depth: int):
    """
    Generate positional encoding for input sequences.

    Args:
        length (int): Length of the input sequence.
        depth (int): Depth of the model.

    Returns:
        ndarray: Positional encoding matrix.
    """

    positions: Union[ndarray[Any, dtype], ndarray[Any, dtype[Union[timedelta64, timedelta64]]]] \
        = arange(length)[:, newaxis]
    depths: Any = arange(depth // 2)[newaxis, :] / (depth // 2)

    angle_rates: Union[float, ndarray[Any, dtype[floating]]] = 1 / (10000 ** depths)
    angle_rads: ndarray[Any, dtype[bool_]] = positions * angle_rates

    pos_encoding: ndarray[Any, dtype] = concatenate(
        [sin(angle_rads), cos(angle_rads)],
        axis=-1
    )
    return cast(pos_encoding, dtype=float32)


@keras.saving.register_keras_serializable(package='PositionalEmbedding')
class PositionalEmbedding(keras.layers.Layer):
    """
    PositionalEmbedding layer adds positional encoding to input embeddings.

    Attributes:
        model_dimensions (int): Dimensionality of the model.
        embedding (keras.layers.Embedding): Embedding layer.
        pos_encoding (ndarray): Positional encoding matrix.
    """

    def __init__(self, vocab_size: int, model_dimensions: int):
        """
        Initializes the PositionalEmbedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            model_dimensions (int): Dimensionality of the model.
        """

        super().__init__()
        self.model_dimensions: int = model_dimensions
        self.embedding: keras.layers.Embedding = keras.layers.Embedding(vocab_size, model_dimensions, mask_zero=True)
        self.pos_encoding = positional_encoding(
            length=JaraConverseModelConfiguration.MAX_POSITIONAL_ENCODING_LENGTH.value, depth=model_dimensions
        )

    def compute_mask(self, *args, **kwargs) -> Optional[Any]:
        """
        Computes the mask for the input.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[Any]: Mask tensor.
        """

        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, inputs, *args, **kwargs) -> Any:
        """
        Forward pass for the PositionalEmbedding layer.

        Args:
            inputs (Any): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Output tensor after adding positional encoding to input embeddings.
        """

        length = shape(inputs)[1]
        inputs_embedded: Any = self.embedding(inputs)
        inputs: Any = inputs_embedded * math.sqrt(cast(self.model_dimensions, float32))
        inputs = inputs + self.pos_encoding[newaxis, :length, :]
        return inputs


if __name__ == '__main__':
    from tensorflow import expand_dims

    embed_title = PositionalEmbedding(
        vocab_size=54000,
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value
    )
    embed_code = PositionalEmbedding(
        vocab_size=54000,
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value
    )
    input_ids = expand_dims(input=[1, 2626, 279, 445, 358, 10194, 2795, 12321, 2887, 1450], axis=0)
    targ_in = expand_dims(input=[1, 536, 10194, 67, 474, 12, 92, 16, 677, 4672], axis=0)

    title_emb = embed_title(input_ids)
    code_emb = embed_code(targ_in)
    print(title_emb._keras_mask)

    pos_enc = positional_encoding(length=2050, depth=128)
    print(pos_enc.shape)

    assert keras.saving.get_registered_object('positional_encoding>positional_encoding') == positional_encoding
    assert keras.saving.get_registered_name(positional_encoding) == 'positional_encoding>positional_encoding'

    assert keras.saving.get_registered_object('PositionalEmbedding>PositionalEmbedding') == PositionalEmbedding
    assert keras.saving.get_registered_name(PositionalEmbedding) == 'PositionalEmbedding>PositionalEmbedding'
