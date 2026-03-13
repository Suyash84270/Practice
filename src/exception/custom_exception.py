import sys
import traceback


class CustomException(Exception):
    """
    Custom exception class for better error tracking in projects.
    Provides detailed error message including file name and line number.
    Reusable across ML, CV, LLM, and backend projects.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self._get_detailed_error_message(error_message, error_detail)

    def _get_detailed_error_message(self, error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        detailed_message = (
            f"Error occurred in script: [{file_name}] "
            f"at line: [{line_number}] "
            f"with message: [{error_message}]"
        )

        return detailed_message

    def __str__(self):
        return self.error_message