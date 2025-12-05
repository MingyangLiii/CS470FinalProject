#!/usr/bin/env python3
"""
日志功能使用示例
演示如何使用utils.py中的Logger类来自动保存打印输出
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import Logger, start_logging, log_function_output

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 方法1: 直接创建Logger实例
    logger = Logger("example_log.txt", auto_save=True)
    
    print("这行文本会同时显示在控制台和保存到日志文件")
    print("当前时间:", pd.Timestamp.now())
    
    # 禁用日志记录
    logger.disable_logging()
    print("这行只会显示在控制台，不会保存到日志文件")
    
    # 重新启用
    logger.enable_logging()
    print("重新启用后的日志记录")


def example_context_manager():
    """上下文管理器使用示例"""
    print("\n=== 上下文管理器示例 ===")
    
    # 使用with语句，自动管理日志记录的开始和结束
    with Logger("context_example.log") as logger:
        print("在上下文管理器内的所有输出都会被记录")
        for i in range(3):
            print(f"循环输出 {i+1}")
    
    print("退出上下文管理器后，输出不再被记录")


@log_function_output("decorator_example.log")
def example_decorator():
    """装饰器使用示例"""
    print("\n=== 装饰器示例 ===")
    print("使用装饰器的函数，所有输出都会自动记录")
    return "函数执行完成"


def example_quick_start():
    """快速启动示例"""
    print("\n=== 快速启动示例 ===")
    
    # 使用便捷函数快速启动
    logger = start_logging("quick_start.log")
    
    print("使用start_logging()快速启动日志记录")
    print("适用于快速测试和调试")


if __name__ == "__main__":
    import pandas as pd
    
    try:
        example_basic_usage()
        example_context_manager()
        
        result = example_decorator()
        print(f"装饰器函数返回: {result}")
        
        example_quick_start()
        
        print("\n=== 所有示例执行完成 ===")
        print("请检查生成的日志文件：")
        print("- example_log.txt")
        print("- context_example.log")
        print("- decorator_example.log")
        print("- quick_start.log")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")