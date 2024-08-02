"""
This module defines a class for visualizing the training history and performance of a conversational model.
"""

from GlobalVariables import VariableParameters
import plotly.graph_objects as go
from os import path, makedirs
from pickle import load
import webbrowser


class JaraConverseVisualizer(object):
    """
    Class for visualizing the training history and performance of a conversational model.

    Attributes:
        _model_history (dict): Dictionary containing the training history of the model.
    """

    def __init__(self):
        """
        Initializes the JaraConverseVisualizer object.
        """

        self._model_history: dict = self.load_model_history()
        self._validate_dir()

    @staticmethod
    def _validate_dir() -> None:
        """
        Validates the directory for saving visualization reports.
        """

        makedirs(VariableParameters.VISUALIZER_DIR.value, exist_ok=True)

    @staticmethod
    def load_model_history() -> dict:
        """
        Loads the model training history from a file.

        Returns:
            dict: Loaded model history.
        """

        with open(VariableParameters.SAVED_HISTORY_PATH.value, 'rb') as file:
            return load(file=file)

    def generate_html_report(self) -> None:
        """
        Loads the model training history from a file.

        Returns:
            dict: Loaded model history.
        """

        fig = go.Figure()
        epochs = list(range(1, len(self._model_history['loss']) + 1))
        fig.add_trace(go.Scatter(x=epochs,
                                 y=self._model_history['loss'],
                                 mode='lines+markers',
                                 name='Training Loss'))

        fig.add_trace(go.Scatter(x=epochs,
                                 y=self._model_history['val_loss'],
                                 mode='lines+markers',
                                 name='Validation Loss'))
        fig.update_layout(title='JaraConverse Training and Validation Loss',
                          xaxis_title='Epochs',
                          yaxis_title='Loss',
                          template='plotly_dark',
                          paper_bgcolor="black",
                          plot_bgcolor='rgba(0, 0, 0, 1)')

        performance_summary = self.evaluate_model_performance()
        fig.add_annotation(text=performance_summary,
                           xref='paper', yref='paper',
                           x=0.5, y=0.9,
                           showarrow=False)

        file_path: str = path.join(VariableParameters.VISUALIZER_DIR.value, 'training_loss_report.html')
        fig.write_html(file_path)
        if not webbrowser.open(url=file_path):
            print("DEBUG: Unable to open loss report please open it manually.")

    def evaluate_model_performance(self) -> str:
        """
        Evaluates the performance of the model based on the validation loss.

        Returns:
            str: Feedback on model performance.
        """

        final_val_loss = self._model_history['val_loss'][-1]
        if final_val_loss < 0.2:
            feedback = "Tip: The model seems to be performing exceptionally well on the validation set. Great job!"
        elif final_val_loss < 0.5:
            feedback = "Tip: The model performance on the validation set is good. Keep refining it for even better " \
                       "results. "
        else:
            feedback = "Tip: The model might benefit from further training or adjustments. Consider reviewing the " \
                       "architecture. "

        return feedback


if __name__ == '__main__':
    JaraConverseVisualizer().generate_html_report()
