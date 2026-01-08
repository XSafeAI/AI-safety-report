"""
DeepSeek模型调用实现
"""

import os
import warnings
from typing import Any, Dict, List, Optional

from openai import OpenAI, AsyncOpenAI

from .base import BaseLLMProvider
from .utils import is_url, is_base64_data_url, is_local_file, local_image_to_base64


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek模型Provider"""
    
    # 支持思考的模型
    REASONING_MODELS = [
        'deepseek-reasoner',
    ]
    
    # 不支持思考的模型
    NON_REASONING_MODELS = [
        'deepseek-chat',
    ]
    
    # 所有支持的模型
    SUPPORTED_MODELS = REASONING_MODELS + NON_REASONING_MODELS
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """
        初始化DeepSeek Provider
        
        Args:
            api_key: DeepSeek API密钥，如果不提供则从环境变量DEEPSEEK_API_KEY读取
            base_url: API基础URL，默认为DeepSeek API端点
            **kwargs: 其他配置参数
        """
        super().__init__(api_key, **kwargs)
        
        # 默认base_url
        if base_url is None:
            base_url = "https://api.deepseek.com"
        
        # 如果提供了api_key，使用提供的key，否则从环境变量DEEPSEEK_API_KEY读取
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=kwargs.get('timeout', 300)
            )
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=kwargs.get('timeout', 300)
            )
        else:
            # 从环境变量读取
            api_key_from_env = os.environ.get("DEEPSEEK_API_KEY")
            self.client = OpenAI(
                api_key=api_key_from_env,
                base_url=base_url,
                timeout=kwargs.get('timeout', 300)
            )
            self.async_client = AsyncOpenAI(
                api_key=api_key_from_env,
                base_url=base_url,
                timeout=kwargs.get('timeout', 300)
            )
    
    def _is_reasoning_model(self, model: str) -> bool:
        """判断是否为支持思考的模型"""
        return any(model.startswith(m) or model == m for m in self.REASONING_MODELS)
    
    def _is_non_reasoning_model(self, model: str) -> bool:
        """判断是否为不支持思考的模型"""
        return any(model.startswith(m) or model == m for m in self.NON_REASONING_MODELS)
    
    def _process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理消息列表，DeepSeek不支持多模态输入，遇到图片或视频会发出警告并跳过
        
        Args:
            messages: 原始消息列表
            
        Returns:
            处理后的消息列表
        """
        processed_messages = []
        has_warned_image = False
        has_warned_video = False
        
        for msg in messages:
            processed_msg = msg.copy()
            content = msg.get('content')
            
            # 如果content是列表（多模态消息）
            if isinstance(content, list):
                processed_content = []
                for item in content:
                    if isinstance(item, dict):
                        # DeepSeek不支持图片输入
                        if item.get('type') == 'image_url':
                            if not has_warned_image:
                                warnings.warn(
                                    "警告: DeepSeek模型不支持图片输入，图片内容将被跳过。",
                                    UserWarning
                                )
                                has_warned_image = True
                            # 跳过图片内容
                            continue
                        
                        # DeepSeek不支持视频输入
                        elif item.get('type') == 'video_url':
                            if not has_warned_video:
                                warnings.warn(
                                    "警告: DeepSeek模型不支持视频输入，视频内容将被跳过。",
                                    UserWarning
                                )
                                has_warned_video = True
                            # 跳过视频内容
                            continue
                        
                        else:
                            # 保留文本等其他类型的内容
                            processed_content.append(item)
                    else:
                        # 保留非字典类型的内容（如字符串）
                        processed_content.append(item)
                
                processed_msg['content'] = processed_content
            
            processed_messages.append(processed_msg)
        
        return processed_messages
    
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_full_response: bool = False,
        video_input_mode: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        DeepSeek模型的completion调用
        
        Args:
            model: 模型名称
            messages: 消息列表
            reasoning_effort: 推理强度（对DeepSeek无效，模型自动决定）
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            return_full_response: 是否返回完整的response对象
            video_input_mode: 视频输入模式（DeepSeek不支持视频输入，此参数无效）
            **kwargs: 其他参数
            
        Returns:
            - 当return_full_response=False时：
                - 如果没有思考内容：返回字符串（输出内容）
                - 如果有思考内容：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回完整的API响应对象
            
        Note:
            - deepseek-reasoner总是会进行思考，无法通过参数控制
            - deepseek-chat不支持思考功能
            - reasoning_effort参数对DeepSeek模型无效
            - DeepSeek不支持视频输入
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # DeepSeek不支持视频输入
        if video_input_mode is not None:
            warnings.warn(
                "警告: DeepSeek模型不支持视频输入，video_input_mode参数无效。",
                UserWarning
            )
        
        # DeepSeek的思考行为是模型内置的，无法通过参数控制
        if reasoning_effort is not None:
            if self._is_reasoning_model(model):
                warnings.warn(
                    f"注意: {model} 模型会自动进行思考，reasoning_effort参数无效。",
                    UserWarning
                )
            elif self._is_non_reasoning_model(model):
                warnings.warn(
                    f"注意: {model} 模型不支持思考功能，reasoning_effort参数将被忽略。",
                    UserWarning
                )
        
        # 处理消息（将本地图片路径转换为base64）
        processed_messages = self._process_messages(messages)
        
        # 构建API调用参数
        api_params = {
            'model': model,
            'messages': processed_messages,
        }
        
        # 添加其他可选参数
        if temperature is not None:
            api_params['temperature'] = temperature
        if max_tokens is not None:
            api_params['max_tokens'] = max_tokens
        if top_p is not None:
            api_params['top_p'] = top_p
        
        # 添加其他额外参数
        api_params.update(kwargs)
        
        # 调用API
        response = self.client.chat.completions.create(**api_params)
        
        # 提取内容
        content = response.choices[0].message.content
        reasoning_content = None
        
        # 如果是reasoning模型，提取思考内容
        if self._is_reasoning_model(model):
            if hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning_content = response.choices[0].message.reasoning_content
        
        # 根据return_full_response决定返回格式
        if return_full_response:
            # 增强response对象
            response.thinking_content = reasoning_content
            return response
        else:
            # 返回解析后的内容
            if reasoning_content:
                return {
                    "content": content,
                    "thinking_content": reasoning_content
                }
            else:
                return content
    
    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_full_response: bool = False,
        video_input_mode: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        DeepSeek模型的异步completion调用
        
        Args:
            model: 模型名称
            messages: 消息列表
            reasoning_effort: 推理强度（对DeepSeek无效，模型自动决定）
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            return_full_response: 是否返回完整的response对象
            video_input_mode: 视频输入模式（DeepSeek不支持视频输入，此参数无效）
            **kwargs: 其他参数
            
        Returns:
            - 当return_full_response=False时：
                - 如果没有思考内容：返回字符串（输出内容）
                - 如果有思考内容：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回完整的API响应对象
            
        Note:
            - deepseek-reasoner总是会进行思考，无法通过参数控制
            - deepseek-chat不支持思考功能
            - reasoning_effort参数对DeepSeek模型无效
            - DeepSeek不支持视频输入
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # DeepSeek不支持视频输入
        if video_input_mode is not None:
            warnings.warn(
                "警告: DeepSeek模型不支持视频输入，video_input_mode参数无效。",
                UserWarning
            )
        
        # DeepSeek的思考行为是模型内置的，无法通过参数控制
        if reasoning_effort is not None:
            if self._is_reasoning_model(model):
                warnings.warn(
                    f"注意: {model} 模型会自动进行思考，reasoning_effort参数无效。",
                    UserWarning
                )
            elif self._is_non_reasoning_model(model):
                warnings.warn(
                    f"注意: {model} 模型不支持思考功能，reasoning_effort参数将被忽略。",
                    UserWarning
                )
        
        # 处理消息（将本地图片路径转换为base64）
        processed_messages = self._process_messages(messages)
        
        # 构建API调用参数
        api_params = {
            'model': model,
            'messages': processed_messages,
        }
        
        # 添加其他可选参数
        if temperature is not None:
            api_params['temperature'] = temperature
        if max_tokens is not None:
            api_params['max_tokens'] = max_tokens
        if top_p is not None:
            api_params['top_p'] = top_p
        
        # 添加其他额外参数
        api_params.update(kwargs)
        
        # 异步调用API
        response = await self.async_client.chat.completions.create(**api_params)
        
        # 提取内容
        content = response.choices[0].message.content
        reasoning_content = None
        
        # 如果是reasoning模型，提取思考内容
        if self._is_reasoning_model(model):
            if hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning_content = response.choices[0].message.reasoning_content
        
        # 根据return_full_response决定返回格式
        if return_full_response:
            # 增强response对象
            response.thinking_content = reasoning_content
            return response
        else:
            # 返回解析后的内容
            if reasoning_content:
                return {
                    "content": content,
                    "thinking_content": reasoning_content
                }
            else:
                return content

