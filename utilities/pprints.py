from platform import system as system_platform
from os.path import isfile
from psutil import Process
from os import system
import time


class PPrints:
    """
    Class for printing formatted logs and statistics during model training.
    This module defines a class `PPrints` for printing formatted logs and statistics during model training.
    Attributes:
        HEADER (str): ANSI escape code for header color.
        BLUE (str): ANSI escape code for blue color.
        CYAN (str): ANSI escape code for cyan color.
        GREEN (str): ANSI escape code for green color.
        WARNING (str): ANSI escape code for warning color.
        RED (str): ANSI escape code for red color.
        RESET (str): ANSI escape code for resetting color to default.
    """

    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

    def __init__(self, logs_file: str = 'logs.txt'):
        """
        Initializes the PPrints object.

        Args:
            logs_file (str): Path to the log file.
        """

        self._process = Process()
        self._log_file = logs_file
        self._start_time = time.time()
        self._platform, self._clear_cmd = self._identify_platform()

    @staticmethod
    def _identify_platform() -> tuple[str, str]:
        """
        Identifies the current platform.

        Returns:
            tuple: A tuple containing platform name and clear command.
        """

        import sys
        if system_platform().lower() == "windows":
            return 'Windows', "cls"
        elif 'google.colab' in sys.modules:
            return "Colab", "nan"
        else:
            return system_platform(), 'clear'

    def clean_terminal(self) -> str:
        """
        Cleans the terminal output.

        Returns:
            str: Platform information.
        """

        if self._platform == "Colab":
            from IPython.display import clear_output
            clear_output()
            return "Google Colab"
        else:
            system(self._clear_cmd)
            return self._platform

    def pretty_print(self, status: str, cpu_limit: str, gpu_limit: str, logs: bool = True) -> None:
        """
        Prints formatted logs and statistics.

        Args:
            status (str): Status message.
            cpu_limit (str): CPU limit information.
            gpu_limit (str): GPU limit information.
            logs (bool): Whether to log the output.
        """

        memory_info = self._process.memory_info()
        elapsed_time = time.time() - self._start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        hours, minutes = divmod(minutes, 60)

        current_memory_usage = memory_info.rss / 1024 / 1024  # Convert bytes to megabytes
        non_log_msg = f"Status => {status}\n" \
                      f"cpu limit => {cpu_limit}\n" \
                      f"gpu limit => {gpu_limit}\n" \
                      f"Execution time => {hours:02d}:{minutes:02d}:{seconds:02d}\n" \
                      "===" * 10 + '\n\n'

        log_msg = f"{self.WARNING}Platform => {self.clean_terminal()}\n" \
                  f"{self.CYAN}Developer => AbdulMoez\n" \
                  f"{self.WARNING}GitHub => github.com/Anonym0usWork1221\n" \
                  f"{self.BLUE}Execution time => {hours:02d}:{minutes:02d}:{seconds:02d}\n" \
                  f"{self.BLUE}Status => {status}\n" \
                  f"{self.CYAN}CPU Limit => {cpu_limit}\n" \
                  f"{self.CYAN}GPU Limit => {gpu_limit}\n" \
                  f"{self.RED}Memory-Usage => {current_memory_usage: .2f}MB\n" \
                  f"{self.RED}Warning => Don't open the output files while script is executing\n{self.RESET}"
        print(log_msg)
        if logs:
            if isfile(self._log_file):
                file_obj = open(self._log_file, "a")
                file_obj.write(non_log_msg)
                file_obj.close()
            else:
                file_obj = open(self._log_file, "w")
                file_obj.write(non_log_msg)
                file_obj.close()
