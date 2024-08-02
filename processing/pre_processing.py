from GlobalVariables import (DataBaseConfiguration, TransformersTokenizerConfiguration, VariableParameters,
                             JaraConverseModelConfiguration, AutoCalculateModelParams)
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizerFast
from tensorflow import int32, TensorShape, data
from typing import Any
import math


class PreProcessing(object):
    """
    PreProcessing class handles the data preprocessing steps required for training a model with TensorFlow and Keras
    using transformers tokenizer.

    Methods:
        load_and_pre_process_dataset() -> DatasetDict:
            Loads and preprocesses the dataset from a SQL database, splits it if required,
            and saves the processed dataset to disk.

        load_cleaned_dataset() -> DatasetDict:
            Loads the cleaned dataset from disk, setting the number of training and validation steps.

        _tokenize_corpus(element: Any, tokenizer: PreTrainedTokenizerFast, max_input_length: int,
                          max_target_length: int) -> dict:
            Tokenizes the input and output data using the specified tokenizer and returns a
            dictionary with tokenized inputs and targets.

        tokenize_code(tokenizer: PreTrainedTokenizerFast, dataset: DatasetDict) -> tuple:
            Tokenizes the dataset and returns TensorFlow datasets for training and validation.

        _extract_features(element: Any) -> tuple:
            Extracts features from the tokenized data to format it for model training.

        _get_train_tf_dataset(train_dataset: Dataset, train_seed: int, training_batch_size: int) -> Any:
            Converts the training dataset to a TensorFlow dataset, shuffling and batching it appropriately.

        _get_validation_tfdataset(eval_dataset: Dataset, validation_batch_size: int) -> Any:
            Converts the validation dataset to a TensorFlow dataset and batches it appropriately.
    """

    @staticmethod
    def load_and_pre_process_dataset() -> DatasetDict:
        """
        Loads and preprocesses the dataset from a SQL database.

        This method loads the dataset from a specified SQL database, removes unnecessary columns,
        splits the dataset into training and validation sets if required, and saves the processed dataset to disk.

        Returns:
           DatasetDict: A dictionary containing the training and validation datasets.
        """

        dataset = Dataset.from_sql(
            sql=DataBaseConfiguration.DATABASE_TABLE_NAME.value,
            con=f"sqlite:///{DataBaseConfiguration.TRAINING_DATABASE_PATH.value}"
        )
        raw_datasets = DatasetDict({"train": dataset})
        if DataBaseConfiguration.UNNECESSARY_COLUMNS_IN_DB.value:
            raw_datasets = raw_datasets.remove_columns(
                column_names=DataBaseConfiguration.UNNECESSARY_COLUMNS_IN_DB.value
            )
        num_of_train_examples = len(raw_datasets['train'])
        num_updates_per_epoch = num_of_train_examples // TransformersTokenizerConfiguration.TRAINING_BATCH_SIZE.value
        AutoCalculateModelParams.STEP_PER_TRAINING_EPOC = num_updates_per_epoch

        if DataBaseConfiguration.SPLIT_DATASET.value:
            raw_datasets = raw_datasets['train'].train_test_split(
                test_size=DataBaseConfiguration.SPLIT_PERCENTAGE.value,
                shuffle=DataBaseConfiguration.SHUFFLE_DATASET.value
            )
            num_of_validation_examples = len(raw_datasets['test'])
            num_test_updates_per_epoch = math.ceil(
                num_of_validation_examples / TransformersTokenizerConfiguration.VALIDATION_BATCH_SIZE.value
            )
            AutoCalculateModelParams.STEP_PER_VALIDATION_EPOC = num_test_updates_per_epoch

        # save dataset
        raw_datasets.save_to_disk(
            dataset_dict_path=VariableParameters.CLEANED_DATASET_DIR.value
        )
        return raw_datasets

    def load_cleaned_dataset(self) -> DatasetDict:
        """
        Loads the cleaned dataset from disk. (If force reloading dataset is enabled it will reload data from database).

        This method attempts to load a preprocessed dataset from disk, setting the number of training
        and validation steps based on the dataset size. If the dataset is not found, it raises a FileNotFoundError.

        Returns:
            DatasetDict: A dictionary containing the training and validation datasets.

        Raises:
            FileNotFoundError: If the cleaned dataset is not found on disk.
        """

        try:
            if DataBaseConfiguration.FORCE_REPROCESS_DATASET.value:
                return self.load_and_pre_process_dataset()
            raw_dataset = load_from_disk(dataset_path=VariableParameters.CLEANED_DATASET_DIR.value)
            num_of_train_examples = len(raw_dataset['train'])
            num_updates_per_epoch = (
                    num_of_train_examples // TransformersTokenizerConfiguration.TRAINING_BATCH_SIZE.value
            )
            AutoCalculateModelParams.STEP_PER_TRAINING_EPOC = num_updates_per_epoch
            if DataBaseConfiguration.SPLIT_DATASET.value:
                num_of_validation_examples = len(raw_dataset['test'])
                num_test_updates_per_epoch = math.ceil(
                    num_of_validation_examples / TransformersTokenizerConfiguration.VALIDATION_BATCH_SIZE.value
                )
                AutoCalculateModelParams.STEP_PER_VALIDATION_EPOC = num_test_updates_per_epoch
            return raw_dataset
        except FileNotFoundError:
            raise FileNotFoundError("Cleaned dataset not found delete the saved states to use default method")

    @staticmethod
    def _tokenize_corpus(element: any, tokenizer: PreTrainedTokenizerFast, max_input_length: int,
                         max_target_length: int) -> dict[str, Any]:
        """
        Tokenizes the input and output data using the specified tokenizer.

        This method tokenizes the 'title' (input) and 'code' (output) fields from the dataset using the provided tokenizer.
        It returns a dictionary containing tokenized inputs and targets.

        Args:
            element (Any): The data element containing 'title' and 'code' fields.
            tokenizer (PreTrainedTokenizerFast): The tokenizer to use for tokenization.
            max_input_length (int): Maximum length for input tokens.
            max_target_length (int): Maximum length for output tokens.

        Returns:
            dict[str, Any]: A dictionary with tokenized 'input_ids', 'targ_in', and 'targ_out'.
        """

        inputs = element['title']
        outputs = element['code']
        model_inputs = tokenizer(inputs, max_length=max_input_length,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="tf").input_ids

        model_outputs = tokenizer(outputs, max_length=max_target_length,
                                  padding="max_length", truncation=True,
                                  return_tensors="tf").input_ids
        targ_in = model_outputs[:, :-1]
        targ_out = model_outputs[:, 1:]

        # tuple [(model_inputs, targ_in), targ_out]
        return {"input_ids": model_inputs, "targ_in": targ_in, "targ_out": targ_out}

    def tokenize_code(self,
                      tokenizer: PreTrainedTokenizerFast,
                      dataset: DatasetDict
                      ) -> tuple:
        """
        Tokenizes the dataset and returns TensorFlow datasets for training and validation.

        This method applies the `_tokenize_corpus` function to the dataset, and then formats the
        tokenized data for training and validation using TensorFlow datasets.

        Args:
            tokenizer (PreTrainedTokenizerFast): The tokenizer to use for tokenization.
            dataset (DatasetDict): The dataset to tokenize.

        Returns:
            tuple: A tuple containing the training and validation TensorFlow datasets.
        """

        dataset = dataset.map(self._tokenize_corpus, batched=True,
                              fn_kwargs={
                                  "tokenizer": tokenizer,
                                  "max_input_length": JaraConverseModelConfiguration.MAX_MODEL_INPUT_SIZE.value,
                                  "max_target_length": JaraConverseModelConfiguration.MAX_MODEL_OUTPUT_SIZE.value
                              })
        train_ds = self._get_train_tf_dataset(
            train_dataset=dataset['train'],
            train_seed=TransformersTokenizerConfiguration.TRAINING_SEED.value,
            training_batch_size=TransformersTokenizerConfiguration.TRAINING_BATCH_SIZE.value
        )
        val_ds = self._get_validation_tfdataset(
            eval_dataset=dataset["test"],
            validation_batch_size=TransformersTokenizerConfiguration.VALIDATION_BATCH_SIZE.value
        )
        train_ds = train_ds.map(self._extract_features).repeat()
        val_ds = val_ds.map(self._extract_features).repeat()
        return train_ds, val_ds

    @staticmethod
    def _extract_features(element: any) -> tuple:
        """
        Extracts features from the tokenized data to format it for model training.

        This method converts the tokenized inputs into a tuple format required for model training.

        Args:
            element (Any): The tokenized data element.

        Returns:
            tuple: A tuple containing (context, targ_in) and targ_out.
        """

        # Convert into (context, targ_in), targ_out (format)
        return (element['input_ids'], element['targ_in']), element['targ_out']

    @staticmethod
    def _get_train_tf_dataset(train_dataset: Dataset, train_seed: int, training_batch_size: int) -> any:
        """
        Converts the training dataset to a TensorFlow dataset.

        This method formats the training dataset, shuffles, batches, and prefetches it to prepare for training.

        Args:
            train_dataset (Dataset): The tokenized training dataset.
            train_seed (int): Seed for shuffling the dataset.
            training_batch_size (int): Batch size for training.

        Returns:
            Any: A TensorFlow dataset for training.
        """

        num_train_examples: int = len(train_dataset)
        train_dataset.set_format(type='tensorflow', columns=['input_ids', 'targ_in', 'targ_out'])
        return_types = {'input_ids': int32, 'targ_in': int32, 'targ_out': int32}
        return_shapes = {
            'input_ids': TensorShape([None]),
            'targ_in': TensorShape([None]),
            'targ_out': TensorShape([None])
        }
        tf_dataset = data.Dataset.from_generator(
            lambda: train_dataset, return_types, return_shapes
        )
        options = data.Options()
        options.experimental_distribute.auto_shard_policy = data.experimental.AutoShardPolicy.OFF
        tf_dataset = tf_dataset.with_options(options)

        ds = (
            tf_dataset.repeat()
            .shuffle(num_train_examples, seed=train_seed)
            .batch(training_batch_size)
            .prefetch(data.AUTOTUNE)
        )
        return ds

    @staticmethod
    def _get_validation_tfdataset(eval_dataset: Dataset, validation_batch_size: int) -> any:
        """
        Converts the validation dataset to a TensorFlow dataset.

        This method formats the validation dataset, batches, and prefetches it to prepare for validation.

        Args:
            eval_dataset (Dataset): The tokenized validation dataset.
            validation_batch_size (int): Batch size for validation.

        Returns:
            Any: A TensorFlow dataset for validation.
        """

        eval_dataset.set_format(type='tensorflow', columns=['input_ids', 'targ_in', 'targ_out'])
        return_types = {'input_ids': int32, 'targ_in': int32, 'targ_out': int32}
        return_shapes = {
            'input_ids': TensorShape([None]),
            'targ_in': TensorShape([None]),
            'targ_out': TensorShape([None])
        }
        tf_dataset = data.Dataset.from_generator(lambda: eval_dataset, return_types, return_shapes)
        options = data.Options()
        options.experimental_distribute.auto_shard_policy = data.experimental.AutoShardPolicy.OFF
        tf_dataset = tf_dataset.with_options(options)

        ds = (
            tf_dataset.repeat()
            .batch(validation_batch_size)
            .prefetch(data.AUTOTUNE)
        )
        return ds
