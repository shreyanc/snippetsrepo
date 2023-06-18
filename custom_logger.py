import logging

class CustomLogger(logging.Logger):
    def __init__(self, name, log_file=None, level=logging.NOTSET):
        super().__init__(name, level)
        self.default_log_file = log_file
        
        # Create a stream handler to print log messages to the terminal
        self.add_stream_handler()

        if self.default_log_file:
            self.add_file_handler(self.default_log_file)

    def add_file_handler(self, log_file):
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)

    def remove_file_handler(self):
        handlers = self.handlers[:]
        for handler in handlers:
            if isinstance(handler, logging.FileHandler):
                self.removeHandler(handler)
                handler.close()

    def remove_file_handler(self):
        handlers = self.handlers[:]
        for handler in handlers:
            if isinstance(handler, logging.FileHandler):
                self.removeHandler(handler)
                handler.close()

    def add_stream_handler(self):
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self.addHandler(stream_handler)

    def remove_stream_handler(self):
        handlers = self.handlers[:]
        for handler in handlers:
            if isinstance(handler, logging.StreamHandler):
                self.removeHandler(handler)
                handler.close()

    def log(self, level, msg, log_to=None, *args, **kwargs):
        if log_to == 'terminal':
            self.remove_file_handler()
            super().log(level, msg, *args, **kwargs)
            self.add_file_handler(self.default_log_file)
        elif log_to:
            self.remove_file_handler()
            self.add_file_handler(log_to)
            super().log(level, msg, *args, **kwargs)
            self.remove_file_handler()
            self.add_file_handler(self.default_log_file)
        else:
            super().log(level, msg, *args, **kwargs)

# USAGE:

# # Initialize the custom logger with the default log file path

# logger = CustomLogger('my_logger', log_file='default.log')
# logger.setLevel(logging.DEBUG)

# logger.log(logging.INFO, "An info message, should be printed to ONLY TERMINAL", log_to='terminal')
# logger.log(logging.INFO, "A different info message, should be printed to TERMINAL and LOG.txt", log_to='log.txt')
# logger.log(logging.ERROR, "An error message, should be printed to TERMINAL and DEFAULT.log")


