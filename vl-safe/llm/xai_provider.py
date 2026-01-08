"""
xAI模型调用实现
"""
import warnings
import os
from typing import Any, Dict, List, Optional

from xai_sdk import Client
from xai_sdk.chat import user, system, assistant, image

from .base import BaseLLMProvider
from .utils import (
    is_url, is_base64_data_url, is_local_file, 
    local_image_to_base64, extract_video_frames_as_base64
)


class XAIProvider(BaseLLMProvider):
    """xAI模型Provider"""
    
    # 支持的模型列表
    SUPPORTED_MODELS = [
        'grok-4-1-fast-reasoning',
        'grok-4-1-fast-non-reasoning',
    ]
    
    # 支持reasoning_effort参数的模型
    REASONING_MODELS = [
        'grok-4-1-fast-reasoning'
    ]
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        初始化xAI Provider
        
        Args:
            api_key: xAI API密钥，如果不提供则从环境变量XAI_API_KEY读取
            **kwargs: 其他配置参数
        """
        super().__init__(api_key, **kwargs)
        
        # 如果提供了api_key，使用提供的key，否则从环境变量XAI_API_KEY读取
        if self.api_key:
            self.client = Client(api_key=self.api_key, timeout=3600)
        else:
            # 从环境变量读取
            api_key_env = os.getenv("XAI_API_KEY")
            if api_key_env:
                self.client = Client(api_key=api_key_env, timeout=3600)
            else:
                raise ValueError("请提供 api_key 或设置环境变量 XAI_API_KEY")
    
    def _supports_reasoning(self, model: str) -> bool:
        """
        判断模型是否支持reasoning_effort参数
        
        Args:
            model: 模型名称
            
        Returns:
            是否支持reasoning_effort
        """
        return model in self.REASONING_MODELS
    
    def _validate_video_input_mode(self, video_input_mode: Optional[str]) -> str:
        """
        验证并标准化视频输入模式
        
        Args:
            video_input_mode: 视频输入模式
            
        Returns:
            标准化的视频输入模式
        """
        if video_input_mode is None:
            return "frames"  # 默认使用frames模式
        
        video_input_mode = video_input_mode.lower()
        
        # xAI只支持frames模式
        if video_input_mode != "frames":
            if video_input_mode == "base64":
                warnings.warn(
                    "警告: xAI不支持视频base64模式，将使用'frames'模式（抽帧）。",
                    UserWarning
                )
            elif video_input_mode == "upload":
                warnings.warn(
                    "警告: xAI不支持'upload'模式，将使用'frames'模式（抽帧）。",
                    UserWarning
                )
            else:
                warnings.warn(
                    f"警告: 未知的视频输入模式'{video_input_mode}'，将使用'frames'模式。",
                    UserWarning
                )
            return "frames"
        
        return video_input_mode
    
    def _convert_messages_to_xai_format(self, messages: List[Dict[str, Any]]) -> List:
        """
        将标准消息格式转换为xAI SDK格式
        
        Args:
            messages: 标准消息列表
            
        Returns:
            xAI SDK格式的消息列表
        """
        xai_messages = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content')
            
            # 如果content是字符串（纯文本消息）
            if isinstance(content, str):
                if role == 'system':
                    xai_messages.append(system(content))
                elif role == 'user':
                    xai_messages.append(user(content))
                elif role == 'assistant':
                    xai_messages.append(assistant(content))
            
            # 如果content是列表（多模态消息）
            elif isinstance(content, list):
                # 收集文本和图片内容
                text_parts = []
                image_parts = []
                
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get('type')
                        
                        # 处理文本
                        if item_type == 'text':
                            text_parts.append(item.get('text', ''))
                        
                        # 处理图片
                        elif item_type == 'image_url':
                            image_url_data = item.get('image_url', {})
                            if isinstance(image_url_data, dict):
                                url = image_url_data.get('url', '')
                                detail = image_url_data.get('detail', 'high')
                            else:
                                url = image_url_data
                                detail = 'high'
                            
                            # 如果是本地文件，转换为base64
                            if is_local_file(url):
                                url = local_image_to_base64(url, include_mime=True)
                            
                            # 添加图片对象
                            image_parts.append(image(image_url=url, detail=detail))
                        
                        # 处理视频：xAI不原生支持视频，需要抽帧转为多图
                        elif item_type == 'video_url':
                            video_data = item.get('video_url') or item.get('video')
                            if isinstance(video_data, dict):
                                video_path = video_data.get('url', '')
                            elif isinstance(video_data, str):
                                video_path = video_data
                            else:
                                video_path = item.get('video', '')
                            
                            # 移除 file:// 前缀（如果有）
                            if video_path.startswith('file://'):
                                video_path = video_path[7:]
                            
                            # 获取抽帧参数
                            fps = item.get('fps', None)
                            max_frames = item.get('max_frames', None)
                            
                            # 如果都没提供，默认fps=1.0
                            if fps is None and max_frames is None:
                                fps = 1.0
                            
                            # 只处理本地文件（URL和base64暂不支持）
                            if is_local_file(video_path):
                                # 抽取视频帧并转换为base64
                                frame_base64_list = extract_video_frames_as_base64(
                                    video_path, 
                                    fps=fps, 
                                    max_frames=max_frames
                                )
                                
                                # 将每一帧作为一个图片对象添加
                                for frame_base64 in frame_base64_list:
                                    image_parts.append(image(image_url=frame_base64, detail='high'))
                            else:
                                # URL或base64格式的视频暂不支持，跳过
                                warnings.warn(
                                    f"xAI不支持视频URL或base64格式，仅支持本地视频文件抽帧。视频 {video_path} 将被跳过。",
                                    UserWarning
                                )
                
                # 组合文本和图片
                combined_text = ' '.join(text_parts) if text_parts else ''
                
                # 根据role创建消息
                if role == 'user':
                    if image_parts:
                        # 如果有图片，将文本和图片一起传入
                        xai_messages.append(user(combined_text, *image_parts))
                    else:
                        # 只有文本
                        xai_messages.append(user(combined_text))
                elif role == 'system':
                    # system消息通常不包含图片
                    xai_messages.append(system(combined_text))
                elif role == 'assistant':
                    xai_messages.append(assistant(combined_text))
        
        return xai_messages
    
    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_full_response: bool = False,
        video_input_mode: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        xAI模型的completion调用
        
        Args:
            model: 模型名称 (如 grok-4-1-fast-reasoning, grok-4-1-fast-non-reasoning)
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            reasoning_effort: 推理强度 ("low", "medium", "high")，仅部分模型支持
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            return_full_response: 是否返回完整的response对象，默认False返回文本
            video_input_mode: 视频输入模式（xAI只支持"frames"模式）
            **kwargs: 其他xAI API参数
            
        Returns:
            - 当return_full_response=False时：
                - 对于普通模型：返回字符串（输出内容）
                - 对于reasoning模型：返回字典 {"content": "...", "reasoning_tokens": ...}
            - 当return_full_response=True时：返回完整的API响应对象
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # 验证并标准化video_input_mode（xAI只支持frames）
        video_input_mode = self._validate_video_input_mode(video_input_mode)
        
        # 转换消息格式
        xai_messages = self._convert_messages_to_xai_format(messages)
        
        # 构建chat参数
        chat_params = {
            'model': model,
            'messages': xai_messages,
        }
        
        # 处理reasoning_effort参数
        if reasoning_effort is not None:
            if self._supports_reasoning(model):
                chat_params['reasoning_effort'] = reasoning_effort
            else:
                warnings.warn(
                    f"警告: 模型 {model} 不支持 reasoning_effort 参数，该参数将被忽略。"
                    f"只有 {', '.join(self.REASONING_MODELS)} 系列模型支持该参数。",
                    UserWarning
                )
        
        # 添加其他可选参数
        if temperature is not None:
            chat_params['temperature'] = temperature
        if max_tokens is not None:
            chat_params['max_tokens'] = max_tokens
        if top_p is not None:
            chat_params['top_p'] = top_p
        
        # 添加其他额外参数
        chat_params.update(kwargs)
        
        # 创建chat并获取响应
        chat = self.client.chat.create(**chat_params)
        response = chat.sample()
        
        # 根据参数决定返回格式
        if return_full_response:
            return response
        else:
            # 返回解析后的文本内容
            result = response.content
            
            # 如果有reasoning_tokens，一并返回
            if hasattr(response, 'usage') and hasattr(response.usage, 'reasoning_tokens'):
                reasoning_tokens = response.usage.reasoning_tokens
                if reasoning_tokens > 0:
                    return {
                        'content': result,
                        'reasoning_tokens': reasoning_tokens
                    }
            
            return result
    
    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        return_full_response: bool = False,
        video_input_mode: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        xAI模型的异步completion调用
        
        注意：xai_sdk 当前版本可能不支持异步操作，这里使用同步调用实现
        
        Args:
            model: 模型名称 (如 grok-4-1-fast-reasoning, grok-4-1-fast-non-reasoning)
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            reasoning_effort: 推理强度 ("low", "medium", "high")，仅部分模型支持
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            return_full_response: 是否返回完整的response对象，默认False返回文本
            video_input_mode: 视频输入模式（xAI只支持"frames"模式）
            **kwargs: 其他xAI API参数
            
        Returns:
            - 当return_full_response=False时：
                - 对于普通模型：返回字符串（输出内容）
                - 对于reasoning模型：返回字典 {"content": "...", "reasoning_tokens": ...}
            - 当return_full_response=True时：返回完整的API响应对象
        """
        # xai_sdk可能不支持异步，直接调用同步版本
        # 在实际的异步环境中，可以使用 asyncio.to_thread 来包装同步调用
        import asyncio
        return await asyncio.to_thread(
            self.completion,
            model=model,
            messages=messages,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            return_full_response=return_full_response,
            video_input_mode=video_input_mode,
            **kwargs
        )

