import pandas as pd
import sys
import os
from datetime import datetime

def load_csv(path):
    df = pd.read_csv(path)
    return df


class Logger:

    def __init__(self, log_file=None, auto_save=True):
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"output_log_{timestamp}.txt"
        
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.log_file_handle = None
        self.enabled = False
        
        if auto_save:
            self.enable_logging()
    
    def enable_logging(self):
        if not self.enabled:
            # 创建日志目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
            
            # 打开日志文件
            self.log_file_handle = open(self.log_file, 'a', encoding='utf-8')
            
            # 创建自定义的输出类
            self.logger = self._create_logger()
            
            # 重定向标准输出
            sys.stdout = self.logger
            self.enabled = True
            
            print(f"=== Logger Start, Log Save To: {self.log_file} ===")
    
    def disable_logging(self):
        if self.enabled:
            sys.stdout = self.original_stdout
            if self.log_file_handle:
                self.log_file_handle.close()
                self.log_file_handle = None
            self.enabled = False
            print("=== Logger End ===")
    
    def _create_logger(self):
        class LoggerOutput:
            def __init__(self, original_stdout, file_handle):
                self.original_stdout = original_stdout
                self.file_handle = file_handle
            
            def write(self, text):
                self.original_stdout.write(text)
                if self.file_handle:
                    self.file_handle.write(text)
                    self.file_handle.flush() 
            
            def flush(self):
                self.original_stdout.flush()
                if self.file_handle:
                    self.file_handle.flush()
        
        return LoggerOutput(self.original_stdout, self.log_file_handle)
    
    def __enter__(self):
        self.enable_logging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 如果有异常发生，可以选择记录到日志文件
        if exc_type is not None:
            print(f"Error: {exc_type.__name__}: {exc_val}")
        self.disable_logging()
    
    def __del__(self):
        if self.enabled:
            self.disable_logging()


def start_logging(log_file=None):
    return Logger(log_file=log_file, auto_save=True)


def log_function_output(log_file=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with Logger(log_file=log_file):
                return func(*args, **kwargs)
        return wrapper
    return decorator



