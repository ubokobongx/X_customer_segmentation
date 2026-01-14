#!/usr/bin/env python
import sys
import os

# FIX: When running directly, add project root to path
# This MUST come BEFORE any imports from src
if __name__ == "__main__" or __package__ is None:
    # Add the parent directory of src to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from src/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# NOW import from src
try:
    from src.logger import logging
except ImportError as e:
    print(f"Import error: {e}")
    print(f"sys.path: {sys.path}")
    print(f"Current dir: {os.getcwd()}")
    raise

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e, sys)
