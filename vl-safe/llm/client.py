"""
统一的LLM客户端
"""

import asyncio
from typing import Any, Dict, List, Optional

from .base import BaseLLMProvider
from .openai_provider import OpenAIProvider
from .ark_provider import ArkProvider
from .dashscope_provider import DashScopeProvider
from .gemini_provider import GeminiProvider
from .deepseek_provider import DeepSeekProvider
from .siliconflow_provider import SiliconFlowProvider
from .xai_provider import XAIProvider


class LLMClient:
    """统一的LLM调用客户端（使用懒加载模式）"""
    
    def __init__(self, **provider_configs):
        """
        初始化LLM客户端
        
        Args:
            **provider_configs: 各个provider的配置
                例如: openai_api_key="xxx", qwen_api_key="yyy"
        
        注意：使用懒加载模式，provider只在第一次使用时才会被初始化
        """
        self.provider_configs = provider_configs
        
        # 使用字典存储已初始化的provider实例（懒加载）
        self._initialized_providers: Dict[str, BaseLLMProvider] = {}
        
        # Provider类型映射表（定义支持的provider类型）
        self._provider_classes = {
            'openai': OpenAIProvider,
            'ark': ArkProvider,
            'dashscope': DashScopeProvider,
            'gemini': GeminiProvider,
            'deepseek': DeepSeekProvider,
            'siliconflow': SiliconFlowProvider,
            'xai': XAIProvider,
        }
        
        # 自动构建模型到provider的映射表
        self._model_to_provider_map = self._build_model_mapping()
    
    def _build_model_mapping(self) -> Dict[str, str]:
        """
        自动构建模型到provider类型的映射表
        从每个provider类获取支持的模型列表，构建映射关系
        
        Returns:
            模型前缀到provider类型的映射字典
        """
        mapping = {}
        for provider_type, provider_class in self._provider_classes.items():
            # 获取该provider支持的模型列表
            supported_models = provider_class.get_supported_models()
            for model in supported_models:
                # 使用模型名作为key，provider类型作为value
                mapping[model] = provider_type
        return mapping
    
    def _get_provider_for_model(self, model: str) -> Optional[str]:
        """
        根据模型名称判断应该使用哪个provider
        使用自动构建的映射表进行查找
        
        Args:
            model: 模型名称
            
        Returns:
            provider类型名称，如果没有找到则返回None
        """
        if model in self._model_to_provider_map:
            return self._model_to_provider_map[model]
        
        return None
    
    def _get_or_create_provider(self, provider_type: str) -> BaseLLMProvider:
        """
        获取或创建provider实例（懒加载）
        
        Args:
            provider_type: provider类型名称
            
        Returns:
            provider实例
        """
        # 如果已经初始化过，直接返回
        if provider_type in self._initialized_providers:
            return self._initialized_providers[provider_type]
        
        # 获取provider类
        provider_class = self._provider_classes.get(provider_type)
        if provider_class is None:
            raise ValueError(f"未知的provider类型: {provider_type}")
        
        # 提取该provider的配置参数
        api_key = self.provider_configs.get(f'{provider_type}_api_key')
        provider_specific_config = {
            k.replace(f'{provider_type}_', ''): v 
            for k, v in self.provider_configs.items() 
            if k.startswith(f'{provider_type}_') and k != f'{provider_type}_api_key'
        }
        
        # 创建provider实例
        provider = provider_class(api_key=api_key, **provider_specific_config)
        
        # 缓存实例
        self._initialized_providers[provider_type] = provider
        
        return provider
    
    def _get_provider(self, model: str) -> Optional[BaseLLMProvider]:
        """
        根据模型名称自动选择对应的provider（懒加载）
        
        Args:
            model: 模型名称
            
        Returns:
            对应的provider，如果没有找到则返回None
        """
        # 判断应该使用哪个provider
        provider_type = self._get_provider_for_model(model)
        if provider_type is None:
            return None
        
        # 懒加载获取或创建provider
        return self._get_or_create_provider(provider_type)
    
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        return_full_response: bool = False,
        video_input_mode: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        统一的completion接口
        
        Args:
            model: 模型名称
            messages: 消息列表
            return_full_response: 是否返回完整的response对象，默认False返回解析后的内容
            video_input_mode: 视频输入模式，可选值：
                - "base64": 视频直接转base64（适用于ark, dashscope）
                - "frames": 视频抽帧为多张图片（适用于openai, ark, dashscope）
                - "upload": 上传文件到服务器（适用于gemini）
                - None: 使用provider的默认行为
            **kwargs: 其他参数（如reasoning_effort, temperature等）
            
        Returns:
            - 当return_full_response=False时：
                - 对于普通模型：返回字符串（输出内容）
                - 对于有思考内容的模型：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回完整的API响应对象
            
        Raises:
            ValueError: 当模型不支持时抛出异常
        """
        provider = self._get_provider(model)
        
        if provider is None:
            raise ValueError(
                f"未找到支持模型 '{model}' 的provider。"
                f"请检查模型名称是否正确。"
            )
        
        return provider.completion(
            model=model, 
            messages=messages, 
            return_full_response=return_full_response,
            video_input_mode=video_input_mode,
            **kwargs
        )
    
    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        return_full_response: bool = False,
        video_input_mode: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        统一的异步completion接口
        
        Args:
            model: 模型名称
            messages: 消息列表
            return_full_response: 是否返回完整的response对象，默认False返回解析后的内容
            video_input_mode: 视频输入模式，可选值：
                - "base64": 视频直接转base64（适用于ark, dashscope）
                - "frames": 视频抽帧为多张图片（适用于openai, ark, dashscope）
                - "upload": 上传文件到服务器（适用于gemini）
                - None: 使用provider的默认行为
            **kwargs: 其他参数（如reasoning_effort, temperature等）
            
        Returns:
            - 当return_full_response=False时：
                - 对于普通模型：返回字符串（输出内容）
                - 对于有思考内容的模型：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回完整的API响应对象
            
        Raises:
            ValueError: 当模型不支持时抛出异常
        """
        provider = self._get_provider(model)
        
        if provider is None:
            raise ValueError(
                f"未找到支持模型 '{model}' 的provider。"
                f"请检查模型名称是否正确。"
            )
        
        return await provider.acompletion(
            model=model, 
            messages=messages, 
            return_full_response=return_full_response,
            video_input_mode=video_input_mode,
            **kwargs
        )


# 创建一个全局默认客户端实例
_default_client = None


def get_default_client() -> LLMClient:
    """获取默认的LLM客户端"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def completion(
    model: str,
    messages: List[Dict[str, str]],
    return_full_response: bool = False,
    video_input_mode: Optional[str] = None,
    **kwargs
) -> Any:
    """
    便捷的completion函数，使用默认客户端
    
    Args:
        model: 模型名称
        messages: 消息列表
        return_full_response: 是否返回完整的response对象，默认False返回解析后的内容
        video_input_mode: 视频输入模式（"base64", "frames", "upload"或None）
        **kwargs: 其他参数
        
    Returns:
        - 当return_full_response=False时：
            - 对于普通模型：返回字符串（输出内容）
            - 对于有思考内容的模型：返回字典 {"content": "...", "thinking_content": "..."}
        - 当return_full_response=True时：返回完整的API响应对象
    """
    client = get_default_client()
    return client.completion(
        model=model, 
        messages=messages, 
        return_full_response=return_full_response,
        video_input_mode=video_input_mode,
        **kwargs
    )


async def acompletion(
    model: str,
    messages: List[Dict[str, str]],
    return_full_response: bool = False,
    video_input_mode: Optional[str] = None,
    **kwargs
) -> Any:
    """
    便捷的异步completion函数，使用默认客户端
    
    Args:
        model: 模型名称
        messages: 消息列表
        return_full_response: 是否返回完整的response对象，默认False返回解析后的内容
        video_input_mode: 视频输入模式（"base64", "frames", "upload"或None）
        **kwargs: 其他参数
        
    Returns:
        - 当return_full_response=False时：
            - 对于普通模型：返回字符串（输出内容）
            - 对于有思考内容的模型：返回字典 {"content": "...", "thinking_content": "..."}
        - 当return_full_response=True时：返回完整的API响应对象
    """
    client = get_default_client()
    return await client.acompletion(
        model=model, 
        messages=messages, 
        return_full_response=return_full_response,
        video_input_mode=video_input_mode,
        **kwargs
    )

