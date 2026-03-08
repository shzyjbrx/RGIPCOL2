"""
logger.py
---------
简洁的双路输出日志器（同时写文件和控制台）。
"""

import os
from datetime import datetime


class Logger:
    """
    Parameters
    ----------
    log_path          : 日志文件路径
    print_to_console  : 是否同时打印到控制台
    """

    def __init__(self, log_path: str, print_to_console: bool = True):
        self.log_path         = log_path
        self.print_to_console = print_to_console

        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # 写入启动标记
        self.log(f"{'='*60}")
        self.log(f"[LOG START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"{'='*60}")

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line      = f"[{timestamp}] {message}"

        # 写文件
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        # 控制台输出
        if self.print_to_console:
            print(line)