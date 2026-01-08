"""
LLM Provider基类定义
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLMProvider(ABC):
    """LLM Provider基类"""
    
    # 子类必须定义支持的模型列表
    SUPPORTED_MODELS: List[str] = []
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        初始化Provider
        
        Args:
            api_key: API密钥，如果不提供则从环境变量读取
            **kwargs: 其他配置参数
        """
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        return_full_response: bool = False,
        **kwargs
    ) -> Any:
        """
        统一的completion接口
        
        Args:
            model: 模型名称
            messages: 消息列表
            return_full_response: 是否返回完整的response对象，默认False返回解析后的文本
            **kwargs: 其他参数
            
        Returns:
            - 当return_full_response=False时：
                - 对于普通模型：返回字符串（输出内容）
                - 对于有思考内容的模型：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回完整的API响应对象
        """
        pass
    
    @abstractmethod
    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        return_full_response: bool = False,
        **kwargs
    ) -> Any:
        """
        统一的异步completion接口
        
        Args:
            model: 模型名称
            messages: 消息列表
            return_full_response: 是否返回完整的response对象，默认False返回解析后的文本
            **kwargs: 其他参数
            
        Returns:
            - 当return_full_response=False时：
                - 对于普通模型：返回字符串（输出内容）
                - 对于有思考内容的模型：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回完整的API响应对象
        """
        pass
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """
        获取支持的模型列表（类方法，可在未实例化时调用）
        
        Returns:
            支持的模型列表
        """
        return cls.SUPPORTED_MODELS
    
    def supports_model(self, model: str) -> bool:
        """
        判断是否支持该模型
        
        Args:
            model: 模型名称
            
        Returns:
            是否支持
        """
        # 支持精确匹配或前缀匹配
        return any(model.startswith(supported) or model == supported 
                   for supported in self.SUPPORTED_MODELS)

