import logging
import sys


class ALogger:
    def __init__(self, name, is_console_handler):
        self.logger = logging.getLogger(name)
        # formatter = logging.Formatter("%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        self.fmt = logging.Formatter("%(filename)s %(funcName)s %(lineno)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        self.logger.setLevel(logging.INFO)
        # logger.removeHandler(file_handler)
        if not self.logger.hasHandlers():
            if is_console_handler:
                self.setConsoleHandler()
            else:
                self.setFileHandler("")

    def getLogger(self):
        return self.logger

    def setFileHandler(self, file_path):
        file_handler = logging.FileHandler("FTEST.log")
        file_handler.setFormatter(self.fmt)
        self.logger.addHandler(file_handler)

    def setConsoleHandler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = self.fmt
        self.logger.addHandler(console_handler)
