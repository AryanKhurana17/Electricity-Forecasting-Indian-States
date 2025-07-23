import sys
from electricityforecasting.logger.logger import logging 

class ElectricityForecastingException(Exception):
    def __init__(self, error_message, error_details: sys):
        """
        Custom exception class for handling errors with line number, file name and custom message.
        """
        super().__init__(error_message)
        self.error_message = error_message

        # Extract traceback details
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

        logging.info(self.__str__())

    def __str__(self):
        return f"Error occurred in python script [{self.file_name}] at line [{self.lineno}]: {self.error_message}"
