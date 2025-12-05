import pandas as pd
import sys
import os
from datetime import datetime

def load_csv(path):
    df = pd.read_csv(path)
    return df


class Logger:
    """自动捕获并保存所有打印输出到日志文件的类"""
    
    def __init__(self, log_file=None, auto_save=True):
        """
        初始化日志记录器
        
        Args:
            log_file (str): 日志文件路径，如果为None则自动生成
            auto_save (bool): 是否自动启用日志记录
        """
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
        """启用日志记录"""
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
            
            print(f"=== 日志记录已启用，输出将保存到: {self.log_file} ===")
    
    def disable_logging(self):
        """禁用日志记录"""
        if self.enabled:
            sys.stdout = self.original_stdout
            if self.log_file_handle:
                self.log_file_handle.close()
                self.log_file_handle = None
            self.enabled = False
            print("=== 日志记录已禁用 ===")
    
    def _create_logger(self):
        """创建自定义的日志记录器类"""
        class LoggerOutput:
            def __init__(self, original_stdout, file_handle):
                self.original_stdout = original_stdout
                self.file_handle = file_handle
            
            def write(self, text):
                # 同时输出到控制台和文件
                self.original_stdout.write(text)
                if self.file_handle:
                    self.file_handle.write(text)
                    self.file_handle.flush()  # 立即刷新到文件
            
            def flush(self):
                # 刷新输出缓冲区
                self.original_stdout.flush()
                if self.file_handle:
                    self.file_handle.flush()
        
        return LoggerOutput(self.original_stdout, self.log_file_handle)
    
    def __enter__(self):
        """上下文管理器支持"""
        self.enable_logging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        # 如果有异常发生，可以选择记录到日志文件
        if exc_type is not None:
            print(f"异常发生: {exc_type.__name__}: {exc_val}")
        self.disable_logging()
    
    def __del__(self):
        """析构函数，确保文件被正确关闭"""
        if self.enabled:
            self.disable_logging()


def start_logging(log_file=None):
    """
    快速启动日志记录的便捷函数
    
    Args:
        log_file (str): 日志文件路径，如果为None则自动生成
    
    Returns:
        Logger: Logger实例
    """
    return Logger(log_file=log_file, auto_save=True)


def log_function_output(log_file=None):
    """
    装饰器：自动记录函数的所有输出到日志文件
    
    Args:
        log_file (str): 日志文件路径，如果为None则自动生成
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with Logger(log_file=log_file):
                return func(*args, **kwargs)
        return wrapper
    return decorator



