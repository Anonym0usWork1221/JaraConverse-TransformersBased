"""
This module defines classes for generating code snippets based on user input using a pre-trained conversational model.
"""

from GlobalVariables import JaraConverseModelConfiguration, VariableParameters
from processing.tokenizer import JaraConverseTokenizer
from layers.JaraConverseModel import JaraConverseModel
from transformers import PreTrainedTokenizerFast
from os import path, environ, listdir, system
from tensorflow import keras
import tensorflow as tf

# turning off machine instructions logs from tensorflow
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class JaraConverseGenerator(tf.Module):
    """
    Class for generating code snippets based on user input using a pre-trained conversational model.

    Attributes:
        tokenizer (PreTrainedTokenizerFast): Pre-trained tokenizer for tokenizing input sentences.
        jara_converse_model (JaraConverseModel): Pre-trained conversational model.

    """

    def __init__(self, tokenizer: PreTrainedTokenizerFast, model: JaraConverseModel) -> None:
        """
        Initializes the JaraConverseGenerator object.

        Args:
            tokenizer (PreTrainedTokenizerFast): Pre-trained tokenizer for tokenizing input sentences.
            model (JaraConverseModel): Pre-trained conversational model.
        """

        super().__init__()
        self.tokenizer: PreTrainedTokenizerFast = tokenizer
        self.jara_converse_model: JaraConverseModel = model

    def __call__(self, sentence: any([str, list[str]]), max_length: int = 100) -> tuple[str, list[str], any]:
        """
        Generates code snippets based on the input sentence.

        Args:
            sentence (str or list[str]): Input sentence or list of sentences.
            max_length (int): Maximum length of the generated code snippet.

        Returns:
            tuple: A tuple containing the generated code snippet, list of tokens, and attention weights.
        """

        local_sentence: any = sentence
        if isinstance(sentence, str):
            local_sentence = [sentence]

        local_sentence: any = self.tokenizer(
            local_sentence,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        ).input_ids

        local_sentence = tf.cast(local_sentence, dtype=tf.int64)
        encoder_input = local_sentence
        start_end = self.tokenizer('', return_tensors="tf").input_ids[0]
        start = tf.cast(start_end[0][tf.newaxis], dtype=tf.int64)  # start tag
        end = tf.cast(start_end[1][tf.newaxis], dtype=tf.int64)  # end tag
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for token_index in tf.range(max_length):  # tf.range() (creates tensors)
            output = tf.transpose(output_array.stack())
            predictions = self.jara_converse_model([encoder_input, output], training=False)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)
            output_array = output_array.write(token_index + 1, predicted_id[0])

            if predicted_id[0] == end:
                break

        output = tf.transpose(output_array.stack())
        words: list[str] = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in
                            tf.squeeze(output).numpy().tolist()]
        results: str = "".join(words)
        self.jara_converse_model([encoder_input, output[:, :-1]], training=False)
        attention_weights: any = self.jara_converse_model.decoder.last_attn_scores
        return results, words, attention_weights

    def temperature_sampling_steps(self, encoder_input, input_ids, temperature) -> any:
        """
        Perform temperature sampling steps.

        Args:
            encoder_input: Encoder input.
            input_ids: Input IDs.
            temperature (float): Sampling temperature.

        Returns:
            tf.Tensor: Predicted IDs.
        """

        predictions = self.jara_converse_model([encoder_input, input_ids], training=False)
        predictions = predictions[:, -1, :] / temperature
        predicted_ids = tf.random.categorical(predictions, num_samples=1, dtype=tf.int32)
        return predicted_ids

    def temperature_sampling(
            self,
            sentence: any([str, list[str]]),
            temperature: float = 1.0,
            max_length: int = 100
    ) -> str:
        """
        Perform temperature sampling.

        Args:
            sentence (str or list[str]): Input sentence or list of sentences.
            temperature (float): Sampling temperature.
            max_length (int): Maximum length of the generated code snippet.

        Returns:
            str: Decoded output.
        """

        if isinstance(sentence, str):
            sentence = [sentence]

        sentence = self.tokenizer(
            sentence,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        ).input_ids

        sentence = tf.cast(sentence, dtype=tf.int64)
        encoder_input = sentence
        start_end = self.tokenizer('', return_tensors="tf").input_ids[0]
        start = tf.cast(start_end[0][tf.newaxis], dtype=tf.int64)  # start tag
        input_ids = tf.fill([1, 1], start)
        output_ids = []

        for _ in range(max_length):
            predicted_ids = self.temperature_sampling_steps(
                encoder_input=encoder_input,
                input_ids=input_ids,
                temperature=temperature
            )
            input_ids = tf.concat([input_ids, tf.cast(predicted_ids, tf.int64)], axis=-1)
            output_ids.append(predicted_ids.numpy()[0, 0])

        decoded_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return decoded_output

    def greedy_decoding_steps(self, encoder_input, input_ids) -> tf.Tensor:
        """
        Perform greedy decoding steps.

        Args:
            encoder_input: Encoder input.
            input_ids: Input IDs.

        Returns:
            tf.Tensor: Predicted IDs.
        """

        predictions = self.jara_converse_model([encoder_input, input_ids], training=False)
        predicted_ids = tf.argmax(predictions[:, -1, :], axis=-1, output_type=tf.int64)
        return predicted_ids

    def greedy_decoding(
            self,
            sentence: any([str, list[str]]),
            max_length: int = 100
    ) -> str:
        """
        Perform greedy decoding.

        Args:
            sentence (str or list[str]): Input sentence or list of sentences.
            max_length (int): Maximum length of the generated code snippet.
        Returns:
            str: Decoded output.
        """

        if isinstance(sentence, str):
            sentence = [sentence]

        sentence = self.tokenizer(
            sentence, max_length=max_length,
            padding="max_length", truncation=True, return_tensors="tf"
        ).input_ids

        sentence = tf.cast(sentence, dtype=tf.int64)
        encoder_input = sentence
        start_end = self.tokenizer('', return_tensors="tf").input_ids[0]
        start = tf.cast(start_end[0][tf.newaxis], dtype=tf.int64)  # start tag
        input_ids = tf.fill([1, 1], start)
        output_ids = []
        for _ in range(max_length):
            predicted_ids = self.greedy_decoding_steps(encoder_input=encoder_input, input_ids=input_ids)
            input_ids = tf.concat([input_ids, predicted_ids[:, tf.newaxis]], axis=-1)
            output_ids.append(predicted_ids.numpy()[0])

        decoded_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return decoded_output


class JaraConverseDemo(object):
    """
    Class for running sample interactions or evaluations with the JaraConverse model.

    Attributes:
       _tokenizer (PreTrainedTokenizerFast): Pre-trained tokenizer for tokenizing input sentences.
       __load_from_checkpoint (bool): Flag to indicate whether to load the model from a checkpoint.
       __model (JaraConverseModel): Pre-trained conversational model.
       __generator (JaraConverseGenerator): Generator instance for generating code snippets.
    """

    def __init__(self, load_from_checkpoint: bool = True):
        """
        Initializes the JaraConverseDemo object.

        Args:
            load_from_checkpoint (bool): Flag to indicate whether to load the model from a checkpoint.

        """

        self._tokenizer: PreTrainedTokenizerFast = JaraConverseTokenizer().load_tokenizer()
        self.__load_from_checkpoint: bool = load_from_checkpoint
        self.__sample_dataset: list[str] = [
            'Write a python function to check whether the elements in a list are same or not.',
            'Write a function to find the depth of a dictionary.',
        ]
        self.__model = None
        self.__generator = None

    def load_model(self) -> JaraConverseModel:
        """
        Loads the JaraConverse model.

        Returns:
            JaraConverseModel: Loaded conversational model.

        # ISSUE: https://github.com/keras-team/keras-core/issues/855
        # NOTE: There is an issue with loading Keras models in TensorFlow versions 2.13 and 2.14

        model = keras.models.load_model(
            filepath=self._model_path
        )

        # so we are going to build model from scratch and just load trained checkpoints (weights) in our case
        """
        if self.__load_from_checkpoint:
            model: JaraConverseModel = JaraConverseModel(
                num_layers=JaraConverseModelConfiguration.NUMBER_OF_LAYERS.value,
                model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value,
                num_heads=JaraConverseModelConfiguration.NUM_OF_HEADS.value,
                feed_forward_dimensions=JaraConverseModelConfiguration.FF_DIMENSION.value,
                input_vocab_size=self._tokenizer.size,
                target_vocab_size=self._tokenizer.size,
                dropout_rate=JaraConverseModelConfiguration.LEARNING_DROPOUT_RATE.value
            )
            check_point = any(VariableParameters.CHECKPOINT_NAME.value in filename for filename in
                              listdir(path.dirname(path.join(
                                  VariableParameters.CHECKPOINT_DIR.value,
                                  VariableParameters.CHECKPOINT_NAME.value
                              ))))
            if not check_point:
                raise FileNotFoundError("Checkpoints are not found please train the model give the weights file")

            status = model.load_weights(path.join(VariableParameters.CHECKPOINT_DIR.value,
                                                  VariableParameters.CHECKPOINT_NAME.value))
            status.expect_partial()
            return model

        return keras.models.load_model(
            filepath=path.join(
                VariableParameters.SAVED_MODEL_DIR.value, VariableParameters.SAVED_MODEL_NAME.value
            ).__str__()
        )

    def sample_run(self, max_length: int = 100, temperature: float = 0.7) -> None:
        """
        Runs sample interactions with the JaraConverse model.

        Args:
            max_length (int): Maximum length of the generated code snippet.
            temperature (float): The temperature on which sample will be generated (Default 0.7)
        """

        if not self.__model:
            model: JaraConverseModel = self.load_model()
            self.__model = model

        if not self.__generator:
            generator: JaraConverseGenerator = JaraConverseGenerator(
                tokenizer=self._tokenizer, model=self.__model
            )
            self.__generator = generator

        while True:
            try:
                print('\n\n')
                prompt = input("Enter Query: ").lower()
                status = self.__custom_prompts(prompt=prompt, max_output_length=max_length, temperature=temperature)
                if status == 0:
                    continue
                elif status == 1:
                    break
                prompt = prompt[: JaraConverseModelConfiguration.MAX_MODEL_INPUT_SIZE.value].strip()
                result, result_tokens, result_attention_weights = self.__generator(
                    prompt,
                    max_length=max_length
                )
                print("===" * 10 + "UserPrompt" + "===" * 10)
                print(prompt)
                print("\n\n" + "===" * 10 + "Predicted" + "===" * 10)
                print(f"Generated Code:\n\n{result.strip()}")
            except KeyboardInterrupt:
                break

    def __execute_samples(
            self, max_output_length, temperature: float = 0.7, temperature_sampling: bool = False
    ) -> None:
        sampler = f"TemperatureSampler on temperature: {temperature}" if temperature_sampling else "GreedySampler"
        for prompt in self.__sample_dataset:
            if not temperature_sampling:
                result, result_tokens, result_attention_weights = self.__generator(
                    prompt,
                    max_length=max_output_length
                )
            else:
                result = self.__generator.greedy_decoding(sentence=prompt, max_length=max_output_length)
            print("===" * 10 + f"SamplePrompt-{sampler}" + "===" * 10)
            print(prompt)
            print("\n\n" + "===" * 10 + "Predicted" + "===" * 10)
            print(f"Generated Code:\n\n{result.strip()}")

    def __custom_prompts(self, prompt: str, max_output_length: int, temperature: float = 0.7) -> int:
        if prompt == ':s':
            self.__execute_samples(
                max_output_length=max_output_length
            )
            return 0
        elif prompt == ':ts':
            self.__execute_samples(
                max_output_length=max_output_length,
                temperature=temperature,
            )
            return 0
        elif prompt == ':q':
            return 1
        elif prompt == ":a":
            print("===" * 10 + "MODEL ARCHITECTURE" + "===" * 10)
            self.__model.summary()
            print('\n\n')
            return 0
        elif prompt == ':p':
            total_params = self.__model.count_params()
            formatted_params = '{:.2f}'.format(total_params / 1e6) + 'M' if total_params >= 1e6 else '{:.2f}'.format(
                total_params / 1e9) + 'B'
            print("===" * 10 + "MODEL PARAMETERS" + "===" * 10)
            print(formatted_params)
            print('\n\n')
            return 0
        elif prompt.startswith(':'):
            system(prompt.replace(':', '').strip())
            return 0
        else:
            return -1

    def evaluation_execution(self, max_length: int = 100, temperature: float = 0.7) -> None:
        """
        Executes evaluation with the JaraConverse model.

        Args:
            max_length (int): Maximum length of the generated code snippet.
            temperature (float): Sampling temperature for temperature-based decoding.
        """

        if not self.__model:
            model: JaraConverseModel = self.load_model()
            self.__model = model

        if not self.__generator:
            generator: JaraConverseGenerator = JaraConverseGenerator(
                tokenizer=self._tokenizer, model=self.__model
            )
            self.__generator = generator

        while True:
            try:
                print('\n\n')
                prompt = input("Enter Query: ").lower()
                status = self.__custom_prompts(prompt=prompt, max_output_length=max_length, temperature=temperature)
                if status == 0:
                    continue
                elif status == 1:
                    break
                prompt = prompt[: JaraConverseModelConfiguration.MAX_MODEL_INPUT_SIZE.value].strip()
                result, result_tokens, result_attention_weights = self.__generator(
                    prompt,
                    max_length=max_length
                )
                print("===" * 10 + "UserPrompt" + "===" * 10)
                print(prompt)
                print("\n\n" + "===" * 10 + "Predicted Using Simple generator" + "===" * 10)
                print(f"Generated Code:\n\n{result.strip()}")
                temperature_sample = self.__generator.temperature_sampling(
                    sentence=prompt, max_length=max_length, temperature=temperature
                )
                print("\n\n" + "===" * 10 + f"Predicted Using Temperature Sampling on temp: {temperature}" + "===" * 10)
                print(f"Generated Code:\n\n{temperature_sample.strip()}")
                greedy_sample = self.__generator.greedy_decoding(sentence=prompt, max_length=max_length)
                print("\n\n" + "===" * 10 + f"Predicted Using Greedy Sampling" + "===" * 10)
                print(f"Generated Code:\n\n{greedy_sample.strip()}")
            except KeyboardInterrupt:
                break


if __name__ == '__main__':
    JaraConverseDemo().sample_run(max_length=100, temperature=0.7)
