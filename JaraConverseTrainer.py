from GlobalVariables import (
    VariableParameters,
    TransformersTokenizerConfiguration,
    JaraConverseModelConfiguration,
    AutoCalculateModelParams,
    DataBaseConfiguration
)
from layers.customized_adam_lr import AdamScheduler, GarbageCollectorCallback, AdamLearningRateTracker
from utilities.losses import masked_loss, masked_accuracy, ModelLosses
from layers.JaraConverseModel import JaraConverseModel
from processing.tokenizer import JaraConverseTokenizer
from processing.pre_processing import PreProcessing
from tensorflow import config, expand_dims, keras
from transformers import PreTrainedTokenizerFast
from os import makedirs, path, environ, listdir
from codecarbon import EmissionsTracker
from utilities.pprints import PPrints
from datasets import DatasetDict
from pickle import dump, load
from datetime import datetime
import time

# NOTE: output file .codecarbon.config is getting override here. Don't add anything in output file.
CARBON_FILE_CONTENT: str = """
[codecarbon]
save_to_file = true
output_dir = {output_dir_name}
output_file=emissions.csv
co2_signal_api_token=script-overwrite
experiment_id = 235b1da5-aaaa-aaaa-aaaa-853681599f2e
log_level = critical
tracking_mode = process
"""


class JaraConverseTrainer(object):
    """
    A class for training the JaraConverse model using Keras, TensorFlow, and Transformers tokenizer.

    Attributes:
        _max_gpu_utilization_in_mbs (int): Maximum GPU utilization in megabytes.
        _saved_states (dict): Dictionary to store various training states.
        _pprints (PPrints): Instance of PPrints for pretty printing.
        _cpu_limits (str): CPU limitation information.
        _gpu_limits (str): GPU limitation information.
        _pre_processing (PreProcessing): Instance of PreProcessing for dataset preprocessing.
        _tokenizer (JaraConverseTokenizer): Instance of JaraConverseTokenizer for tokenization.
        __co2_emission_tracker (EmissionsTracker): Instance of EmissionsTracker for carbon emissions tracking.
        __tensorboard_logs_dir (str): Directory path for TensorBoard logs.

    Methods:
        _validating_directories(): Creates necessary directories if not already existent.
        _set_gpu_memory_limit(): Sets GPU memory limit.
        _load_saved_states(): Loads saved training states from file.
        _dump_saved_states(): Dumps current training states to file.
        _set_up_config(): Sets up CPU and GPU configuration.
        _create_model_callbacks(): Creates callbacks for the training process.
        train_model(): Trains the JaraConverse model.
    """

    def __init__(self) -> None:
        """
        Initializes the JaraConverseTrainer class.
        """

        self._max_gpu_utilization_in_mbs: int = VariableParameters.MAX_GPU_UTILIZATION_ON_LIMIT.value * 1024
        self._saved_states: dict = {
            'is_dataset_cleaned': False,
            'tokenizer_training': TransformersTokenizerConfiguration.TRAIN_TOKENIZER.value,
            'is_model_training_completed': False,
        }
        self._pprints: PPrints = PPrints()
        self._cpu_limits: str = (f"OMP: {VariableParameters.OMP_THREADS.value}  "
                                 f"MKL: {VariableParameters.MKL_THREADS.value}   "
                                 f"Parallelism Threads: "
                                 f"{VariableParameters.INTER_AND_INTRA_OP_PARALLELISM_THREADS.value}") \
            if VariableParameters.SET_LIMIT_ON_CPU.value else "CPU limits is not set"
        self._gpu_limits: str = f"Max GBS: {VariableParameters.MAX_GPU_UTILIZATION_ON_LIMIT.value}" \
            if VariableParameters.SET_LIMIT_ON_GPU.value else "No limits on GPU"
        self._pre_processing: PreProcessing = PreProcessing()
        self._tokenizer: JaraConverseTokenizer = JaraConverseTokenizer()
        self._validating_directories()
        self.__co2_emission_tracker: EmissionsTracker = EmissionsTracker(
            project_name=VariableParameters.MODEL_NAME.value
        )
        self.__tensorboard_logs_dir = path.join(
            VariableParameters.TENSORBOARD_DIR.value, datetime.now().strftime("%Y%m%d-%H%M%S")
        ).__str__()
        self._load_saved_states()

    @staticmethod
    def _validating_directories() -> None:
        """
        Creates necessary directories if not already existent.

        This method checks if the required directories for saving checkpoints, model states,
        and other data exist. If they don't, it creates them.
        """

        global CARBON_FILE_CONTENT
        makedirs(name=VariableParameters.CHECKPOINT_DIR.value, exist_ok=True)
        makedirs(name=VariableParameters.SAVED_STATES_DIR.value, exist_ok=True)
        makedirs(name=VariableParameters.CLEANED_DATASET_DIR.value, exist_ok=True)
        makedirs(name=VariableParameters.SAVED_MODEL_DIR.value, exist_ok=True)
        makedirs(name=VariableParameters.SAVED_MODEL_WEIGHTS_DIR.value, exist_ok=True)
        makedirs(name=VariableParameters.VISUALIZER_DIR.value, exist_ok=True)
        makedirs(name=VariableParameters.TENSORBOARD_DIR.value, exist_ok=True)

        with open(file=path.join(VariableParameters.BASE_PATH.value, ".codecarbon.config"), mode="w") as carbon_file:
            carbon_file.write(
                CARBON_FILE_CONTENT.format(
                    output_dir_name=VariableParameters.VISUALIZER_DIR.value
                ).strip()
            )
            carbon_file.close()

    def __print_override(self, status: str, logs: bool = False) -> None:
        """
        Handle printing statements more efficiently.
        """
        self._pprints.pretty_print(status=status, logs=logs,
                                   cpu_limit=self._cpu_limits,
                                   gpu_limit=self._gpu_limits)

    def _set_gpu_memory_limit(self) -> None:
        """
        Sets GPU memory limit.

        This method sets the memory limit for the available GPUs to optimize memory usage.
        """

        gpus: any = config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    config.experimental.set_virtual_device_configuration(
                        device=gpu,
                        logical_devices=[config.experimental.VirtualDeviceConfiguration(
                            memory_limit=self._max_gpu_utilization_in_mbs
                        )]
                    )
                config.experimental.list_logical_devices('GPU')
            except RuntimeError as e:
                print(f'Runtime error in setting gpu limits: {e}')

    def _load_saved_states(self) -> None:
        """
        Loads saved training states from file.

        This method loads the previously saved training states from a file into the
        `_saved_states` attribute.
        """

        saved_states_path: str = path.join(
            VariableParameters.SAVED_STATES_DIR.value,
            VariableParameters.SAVED_STATES_NAME.value
        ).__str__()
        if path.isfile(saved_states_path):
            with open(saved_states_path, "rb") as file_pi:
                self._saved_states = load(file_pi)
        else:
            self._dump_saved_states()

    def _dump_saved_states(self) -> None:
        """
        Dumps current training states to file.

        This method saves the current training states stored in the `_saved_states` attribute
        to a file for future use.
        """

        saved_states_path: str = path.join(
            VariableParameters.SAVED_STATES_DIR.value,
            VariableParameters.SAVED_STATES_NAME.value
        ).__str__()
        with open(saved_states_path, 'wb') as file_pi:
            dump(self._saved_states, file_pi)

    def _set_up_config(self) -> None:
        """
        Sets up CPU and GPU configuration.

        This method configures CPU and GPU settings based on the specified parameters,
        such as the number of threads and memory limits.
        """

        if VariableParameters.SET_LIMIT_ON_CPU.value:
            environ["OMP_NUM_THREADS"] = str(VariableParameters.OMP_THREADS.value)
            environ["TF_NUM_INTRAOP_THREADS"] = str(VariableParameters.INTER_AND_INTRA_OP_PARALLELISM_THREADS.value)
            environ["TF_NUM_INTEROP_THREADS"] = str(VariableParameters.INTER_AND_INTRA_OP_PARALLELISM_THREADS.value)
            config.threading.set_inter_op_parallelism_threads(
                VariableParameters.INTER_AND_INTRA_OP_PARALLELISM_THREADS.value
            )
            config.threading.set_intra_op_parallelism_threads(
                VariableParameters.INTER_AND_INTRA_OP_PARALLELISM_THREADS.value
            )
            config.set_soft_device_placement(True)
        devices = config.experimental.list_physical_devices('GPU')
        if devices:
            if VariableParameters.SET_LIMIT_ON_GPU.value:
                self._set_gpu_memory_limit()
        else:
            config.experimental.set_visible_devices(devices=[], device_type='GPU')

    def _create_model_callbacks(self) -> tuple[
        keras.callbacks.ModelCheckpoint,
        keras.callbacks.EarlyStopping,
        AdamScheduler,
        GarbageCollectorCallback,
        AdamLearningRateTracker,
        keras.callbacks.TensorBoard
    ]:
        """
        Creates callbacks for the training process.

        Returns:
           tuple: Tuple containing various callbacks.

        This method creates and configures callbacks to be used during the training process.
        These callbacks include model checkpointing, early stopping, learning rate scheduling,
        and TensorBoard logging.
        """

        checkpoint_path: str = path.join(VariableParameters.CHECKPOINT_DIR.value,
                                         VariableParameters.CHECKPOINT_NAME.value).__str__()
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                              save_best_only=True,
                                                              save_weights_only=True,
                                                              verbose=1)
        model_early_stopping_callback = keras.callbacks.EarlyStopping(
            patience=JaraConverseModelConfiguration.MODEL_EARLY_STOPPING_PATIENCE.value
        )
        adam_scheduler: AdamScheduler = AdamScheduler(
            model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value,
            warmup_steps=JaraConverseModelConfiguration.ADAM_SCHEDULER_WARMUP_STEPS.value
        )
        garbage_cleaner: GarbageCollectorCallback = GarbageCollectorCallback()
        learning_rate_tracker: AdamLearningRateTracker = AdamLearningRateTracker()
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.__tensorboard_logs_dir, histogram_freq=1)
        return (checkpoint_callback,
                model_early_stopping_callback,
                adam_scheduler,
                garbage_cleaner,
                learning_rate_tracker,
                tensorboard_callback)

    def train_model(self) -> None:
        """
        Trains the JaraConverse model.

        This method orchestrates the entire training process for the JaraConverse model.
        It performs dataset preprocessing, model construction, compilation, training,
        and saving of the trained model and its history.
        """

        self.__co2_emission_tracker.start()
        # Step 0: Preprocessing -> Loading callbacks and schedulers
        (checkpoint_callback,
         model_early_stopping_callback,
         adam_scheduler,
         garbage_cleaner,
         learning_rate_tracker,
         tensorboard_callback
         ) = self._create_model_callbacks()
        if JaraConverseModelConfiguration.IS_FIXED_LEARNING_RATE.value:
            adam_optimizer = keras.optimizers.Adam(
                learning_rate=JaraConverseModelConfiguration.FIXED_LEARNING_RATE.value,
                beta_1=JaraConverseModelConfiguration.ADAM_OPTIMIZER_BETA_1.value,
                beta_2=JaraConverseModelConfiguration.ADAM_OPTIMIZER_BETA_2.value,
                epsilon=JaraConverseModelConfiguration.ADAM_OPTIMIZER_EPSILON.value
            )
        else:
            adam_optimizer = keras.optimizers.Adam(
                learning_rate=adam_scheduler,
                beta_1=JaraConverseModelConfiguration.ADAM_OPTIMIZER_BETA_1.value,
                beta_2=JaraConverseModelConfiguration.ADAM_OPTIMIZER_BETA_2.value,
                epsilon=JaraConverseModelConfiguration.ADAM_OPTIMIZER_EPSILON.value
            )

        # Step 1: Clean and save the dataset also create train and test splits.
        if not self._saved_states['is_dataset_cleaned']:
            self.__print_override(status='Loading dataset from sqlite dataset file and preprocess it')
            dataset: DatasetDict = self._pre_processing.load_and_pre_process_dataset()
            self._saved_states['is_dataset_cleaned'] = True
            self._dump_saved_states()
        else:
            if DataBaseConfiguration.FORCE_REPROCESS_DATASET.value:
                self.__print_override(
                    status='Force reprocess dataset is enabled reloading dataset from sqlite database'
                )
            else:
                self.__print_override(status='Loading cleaned dataset from disk')
            dataset: DatasetDict = self._pre_processing.load_cleaned_dataset()

        # Step 2: Load the tokenizer if tokenizer training enables then train the tokenizer and save it
        self.__print_override(status='Loading fast tokenizer')
        tokenizer: PreTrainedTokenizerFast = self._tokenizer.load_tokenizer()
        if TransformersTokenizerConfiguration.TRAIN_TOKENIZER.value:
            self.__print_override(status='Training tokenizer on new dataset it will take a while depend on dataset')
            if not self._saved_states['tokenizer_training']:
                tokenizer: PreTrainedTokenizerFast = self._tokenizer.train_tokenizer_on_new_dataset(
                    pre_trained_tokenizer=tokenizer,
                    dataset=dataset)
                self._saved_states['tokenizer_training'] = True
                self._dump_saved_states()

        # Step 3: Generate tokens ids using tokenizer
        self.__print_override(status='Creating tokenized ids.')
        training_dataset, validation_dataset = self._pre_processing.tokenize_code(tokenizer=tokenizer, dataset=dataset)

        # Step 4: Create and compile model
        converse_model = JaraConverseModel(
            num_layers=JaraConverseModelConfiguration.NUMBER_OF_LAYERS.value,
            model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value,
            num_heads=JaraConverseModelConfiguration.NUM_OF_HEADS.value,
            feed_forward_dimensions=JaraConverseModelConfiguration.FF_DIMENSION.value,
            input_vocab_size=tokenizer.size,
            target_vocab_size=tokenizer.size,
            dropout_rate=JaraConverseModelConfiguration.LEARNING_DROPOUT_RATE.value
        )
        input_ids = expand_dims(input=[1, 2626, 279, 445, 358, 10194, 2795, 12321, 2887, 1450], axis=0)
        targ_in = expand_dims(input=[1, 536, 10194, 67, 474, 12, 92, 16, 677, 4672], axis=0)
        dummy_inputs = (input_ids, targ_in)
        converse_model(dummy_inputs)
        converse_model.compile(loss=masked_loss, optimizer=adam_optimizer, metrics=[masked_accuracy])

        # Step 5: Load the pretrained checkpoints if available
        check_point = any(VariableParameters.CHECKPOINT_NAME.value in filename for filename in
                          listdir(path.dirname(path.join(
                              VariableParameters.CHECKPOINT_DIR.value,
                              VariableParameters.CHECKPOINT_NAME.value
                          ))))
        if check_point:
            self.__print_override(status='====> Loading old Weights <====')
            time.sleep(2)
            converse_model.load_weights(path.join(VariableParameters.CHECKPOINT_DIR.value,
                                                  VariableParameters.CHECKPOINT_NAME.value))

        # Step 6: Train our model on dataset provided to it
        self.__print_override(
            status=f'====> Training Model on {JaraConverseModelConfiguration.MODEL_EPOCHS.value} epochs <===='
        )

        jara_converse_history = converse_model.fit(
            training_dataset,
            epochs=JaraConverseModelConfiguration.MODEL_EPOCHS.value,
            steps_per_epoch=AutoCalculateModelParams.STEP_PER_TRAINING_EPOC,
            validation_data=validation_dataset,
            validation_steps=AutoCalculateModelParams.STEP_PER_VALIDATION_EPOC,
            callbacks=[
                model_early_stopping_callback,
                checkpoint_callback,
                garbage_cleaner,
                learning_rate_tracker,
                tensorboard_callback

            ]
        )

        # Step 7: Saving our model and its train history
        self.__print_override(status='Saving model history and packing model in keras package with weights dump')
        ModelLosses().save_model_history(history=jara_converse_history)
        converse_model.save(path.join(VariableParameters.SAVED_MODEL_DIR.value,
                                      VariableParameters.SAVED_MODEL_NAME.value))
        saved_weight_path: str = path.join(VariableParameters.SAVED_MODEL_WEIGHTS_DIR.value,
                                           VariableParameters.SAVED_MODEL_WEIGHTS_NAME.value).__str__()
        converse_model.save_weights(
            filepath=saved_weight_path
        )
        self.__co2_emission_tracker.stop()
        total_params = converse_model.count_params()
        formatted_params = '{:.2f}'.format(total_params / 1e6) + 'M' if total_params >= 1e6 else '{:.2f}'.format(
            total_params / 1e9) + 'B'
        self.__print_override(status=f'Training complete. Saved\n'
                                     f'Model Saved at => {VariableParameters.SAVED_MODEL_DIR.value}\n'
                                     f'Final learning rate => {converse_model.optimizer.lr.numpy()}\n'
                                     f'Training stopped at epoch => {model_early_stopping_callback.stopped_epoch + 1}\n'
                                     f'Last Validation Loss => {jara_converse_history.history["val_loss"][-1]}\n'
                                     f'Last Validation Accuracy => '
                                     f'{jara_converse_history.history["val_masked_accuracy"][-1]}\n'
                                     f'Total parameters: {formatted_params}'
                              )


if __name__ == '__main__':
    JaraConverseTrainer().train_model()
