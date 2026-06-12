import sys
import os
from loguru import logger

# 标记是否已完成全局配置，避免重复添加 handler
_configured = False


def get_logger():
    """
    返回经过统一配置的 logger 实例。

    日志配置：
    - 控制台输出：INFO 级别及以上
    - 文件输出：logs/app_YYYY-MM-DD.log，INFO 级别及以上
      - 单文件达到 100MB 自动切分
      - 仅保留最近 7 天日志
      - 旧日志自动压缩为 zip
      - 多进程安全（enqueue=True）

    Returns:
        loguru.Logger: 配置好的 logger 实例
    """
    global _configured
    if not _configured:
        # 移除默认的 stderr handler
        logger.remove()

        # 添加控制台输出（彩色，INFO 及以上）
        logger.add(
            sys.stderr,
            level="INFO",
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>"
        )

        # 确保日志目录存在
        os.makedirs("logs", exist_ok=True)

        # 添加文件输出
        logger.add(
            "logs/app_{time:YYYY-MM-DD}.log",
            level="INFO",
            rotation="100 MB",       # 文件达到 100MB 自动切分
            retention="7 days",      # 仅保留 7 天日志
            compression="zip",       # 旧日志自动压缩
            encoding="utf-8",
            enqueue=True,            # 多进程安全
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}"
        )

        _configured = True

    return logger
