"""
统一的LLM调用库
支持多种模型的统一调用接口
"""

from .client import LLMClient, completion, acompletion

__all__ = ['LLMClient', 'completion', 'acompletion']

