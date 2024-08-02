from GlobalVariables import TransformersTokenizerConfiguration
from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict


class JaraConverseTokenizer(object):
    """
    JaraConverseTokenizer class manages the tokenizer used in the JaraConverse model.

    This class handles loading a pre-trained tokenizer and training a new tokenizer on a provided dataset.

    Attributes:
        _tokenizer_path (str): The path to the tokenizer configuration.

    Methods:
        load_tokenizer() -> PreTrainedTokenizerFast:
            Loads the pre-trained tokenizer from the specified path.

        _get_training_corpus(dataset: DatasetDict):
            Retrieves the training corpus from the dataset for tokenizer training.

        train_tokenizer_on_new_dataset(pre_trained_tokenizer: PreTrainedTokenizerFast,
                                       dataset: DatasetDict) -> PreTrainedTokenizerFast:
            Trains a new tokenizer on the provided dataset using the pre-trained tokenizer as a base.
    """

    def __init__(self) -> None:
        """
        Initializes the JaraConverseTokenizer instance.

        Sets the path to the tokenizer configuration from the global configuration.
        """

        self._tokenizer_path: str = TransformersTokenizerConfiguration.TOKENIZER_PATH.value

    def load_tokenizer(self) -> PreTrainedTokenizerFast:
        """
        Loads the pre-trained tokenizer from the specified path.

        This method loads a pre-trained tokenizer from the given path and sets its size attribute based on the
        vocabulary size.

        Returns:
            PreTrainedTokenizerFast: The loaded pre-trained tokenizer.
        """

        loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path=self._tokenizer_path)
        loaded_tokenizer.size = len(loaded_tokenizer.vocab)
        return loaded_tokenizer

    @staticmethod
    def _get_training_corpus(dataset: DatasetDict):
        """
        Retrieves the training corpus from the dataset for tokenizer training.

        This method generates the training corpus in chunks of 1000 examples from the specified dataset.

        Args:
            dataset (DatasetDict): The dataset containing the training data.

        Returns:
            Generator: A generator that yields chunks of the training data for tokenizer training.
        """

        return (
            dataset["train"][i: i + 1000][TransformersTokenizerConfiguration.TRAINING_TOKENIZER_DATA_COLUMN.value]
            for i in range(0, len(dataset["train"]), 1000)
        )

    def train_tokenizer_on_new_dataset(
            self,
            pre_trained_tokenizer: PreTrainedTokenizerFast,
            dataset: DatasetDict
    ) -> PreTrainedTokenizerFast:
        """
        Trains a new tokenizer on the provided dataset using the pre-trained tokenizer as a base.

        This method trains a new tokenizer on the dataset's training corpus, sets the new tokenizers size,
        saves it to the specified path, and returns the trained tokenizer.

        Args:
            pre_trained_tokenizer (PreTrainedTokenizerFast): The pre-trained tokenizer to be used as a base for training.
            dataset (DatasetDict): The dataset containing the training data.

        Returns:
            PreTrainedTokenizerFast: The newly trained tokenizer.
        """

        training_corpus = self._get_training_corpus(dataset=dataset)
        trained_tokenizer = pre_trained_tokenizer.train_new_from_iterator(
            text_iterator=training_corpus,
            vocab_size=TransformersTokenizerConfiguration.TRAINING_TOKENIZER_VOCAB_SIZE.value
        )
        trained_tokenizer.size = len(trained_tokenizer.vocab)
        trained_tokenizer.save_pretrained(self._tokenizer_path)
        return trained_tokenizer
