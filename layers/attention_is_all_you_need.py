"""
Resources: https://arxiv.org/abs/1706.03762
"""

from tensorflow import keras
from typing import Any


@keras.saving.register_keras_serializable(package='BaseAttention')
class BaseAttention(keras.layers.Layer):
    """
    BaseAttention class serves as the base class for different types of attention mechanisms.

    This class initializes the components shared by all attention mechanisms, including the multi-head attention layer,
    normalization layer, and addition layer.

    Attributes:
        multi_head_attention_layer (keras.layers.MultiHeadAttention): Multi-head attention layer.
        normalization_layer (keras.layers.LayerNormalization): Layer normalization layer.
        add_layer (keras.layers.Add): Layer that adds inputs and attention outputs.
        last_attn_scores (Any): Stores the attention scores from the last attention operation.
    """

    def __init__(self, **kwargs):
        """
        Initializes the BaseAttention instance.

        Args:
            **kwargs: Additional keyword arguments for the MultiHeadAttention layer.
        """

        super().__init__()
        self.multi_head_attention_layer: keras.layers.MultiHeadAttention = keras.layers.MultiHeadAttention(**kwargs)
        self.normalization_layer: keras.layers.LayerNormalization = keras.layers.LayerNormalization(
            epsilon=1e-7
        )
        self.add_layer: keras.layers.Add = keras.layers.Add()
        self.last_attn_scores: Any = None


@keras.saving.register_keras_serializable(package='CrossAttention')
class CrossAttention(BaseAttention):
    """
    CrossAttention class implements the cross-attention mechanism, inheriting from BaseAttention.

    This class performs attention over a context sequence (e.g., encoder outputs) given an input sequence
    (e.g., decoder inputs).

    Methods:
        call(inputs: Any, *args, **kwargs) -> Any: Performs cross-attention over the context sequence.
    """

    def call(self, inputs: Any, *args, **kwargs) -> Any:
        """
        Performs cross-attention over the context sequence.

        Args:
            inputs (Any): The input sequence (e.g., decoder inputs).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The output sequence after applying cross-attention.
        """

        context: Any = args[0]
        attn_output, attn_scores = self.multi_head_attention_layer(
            query=inputs,
            key=context,
            value=context,
            return_attention_scores=True)
        self.last_attn_scores: Any = attn_scores
        inputs = self.add_layer([inputs, attn_output])
        inputs = self.normalization_layer(inputs)
        return inputs


@keras.saving.register_keras_serializable(package='GlobalSelfAttention')
class GlobalSelfAttention(BaseAttention):
    """
    GlobalSelfAttention class implements the global self-attention mechanism, inheriting from BaseAttention.

    This class performs self-attention over the entire input sequence.

    Methods:
        call(inputs: Any, *args, **kwargs) -> Any: Performs global self-attention over the input sequence.
    """

    def call(self, inputs: Any, *args, **kwargs) -> Any:
        """
        Performs global self-attention over the input sequence.

        Args:
            inputs (Any): The input sequence.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The output sequence after applying global self-attention.
        """

        attn_output: Any = self.multi_head_attention_layer(
            query=inputs,
            value=inputs,
            key=inputs)
        inputs = self.add_layer([inputs, attn_output])
        inputs = self.normalization_layer(inputs)
        return inputs


@keras.saving.register_keras_serializable(package='CausalSelfAttention')
class CausalSelfAttention(BaseAttention):
    """
    CausalSelfAttention class implements the causal self-attention mechanism, inheriting from BaseAttention.

    This class performs self-attention over the input sequence with a causal mask to prevent attending to future tokens.

    Methods:
        call(inputs: Any, *args, **kwargs) -> Any: Performs causal self-attention over the input sequence.
    """

    def call(self, inputs: Any, *args, **kwargs) -> Any:
        """
        Performs causal self-attention over the input sequence.

        Args:
            inputs (Any): The input sequence.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The output sequence after applying causal self-attention.
        """

        attn_output: Any = self.multi_head_attention_layer(
            query=inputs,
            value=inputs,
            key=inputs,
            use_causal_mask=True)
        inputs = self.add_layer([inputs, attn_output])
        inputs = self.normalization_layer(inputs)
        return inputs


if __name__ == '__main__':
    from tensorflow import expand_dims
    from GlobalVariables import JaraConverseModelConfiguration
    from positional_encoding_and_embeddings import PositionalEmbedding

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

    print(f"Title Embedded Shape: {title_emb.shape}")
    print(f"Code Embedded Shape: {code_emb.shape}")
    print("\n===CrossAttention===")
    sample_ca = CrossAttention(num_heads=2, key_dim=512)
    print(sample_ca(code_emb, title_emb).shape)

    print("\n===GlobalSelfAttention (title_emb only)===")
    sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)
    print(sample_gsa(title_emb).shape)

    print("\n===CausalSelfAttention (code_emb only)===")
    sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)
    print(sample_csa(code_emb).shape)

    assert keras.saving.get_registered_object('BaseAttention>BaseAttention') == BaseAttention
    assert keras.saving.get_registered_name(BaseAttention) == 'BaseAttention>BaseAttention'

    assert keras.saving.get_registered_object('CrossAttention>CrossAttention') == CrossAttention
    assert keras.saving.get_registered_name(CrossAttention) == 'CrossAttention>CrossAttention'

    assert keras.saving.get_registered_object('GlobalSelfAttention>GlobalSelfAttention') == GlobalSelfAttention
    assert keras.saving.get_registered_name(GlobalSelfAttention) == 'GlobalSelfAttention>GlobalSelfAttention'

    assert keras.saving.get_registered_object('CausalSelfAttention>CausalSelfAttention') == CausalSelfAttention
    assert keras.saving.get_registered_name(CausalSelfAttention) == 'CausalSelfAttention>CausalSelfAttention'
