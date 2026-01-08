"""
OpenAI模型调用实现
"""
import warnings
from typing import Any, Dict, List, Optional

from openai import OpenAI, AsyncOpenAI

from .base import BaseLLMProvider
from .utils import (
    is_url, is_base64_data_url, is_local_file, 
    local_image_to_base64, extract_video_frames_as_base64
)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI模型Provider"""
    
    # 支持的模型列表
    SUPPORTED_MODELS = [
        'gpt-5.2',
        'gpt-5.1',
        'gpt-5',
        'gpt-5-mini',
        'gpt-5-nano',
        'gpt-4.1',
        'gpt-4.1-mini',
        'gpt-4.1-nano',
        'gpt-4o',
        'gpt-4o-mini',
    ]
    
    # 支持reasoning_effort参数的模型
    REASONING_MODELS = [
        'gpt-5.2',
        'gpt-5.1',
        'gpt-5',
        'gpt-5-mini',
        'gpt-5-nano'
    ]
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        初始化OpenAI Provider
        
        Args:
            api_key: OpenAI API密钥，如果不提供则从环境变量OPENAI_API_KEY读取
            **kwargs: 其他配置参数
        """
        super().__init__(api_key, **kwargs)
        
        # 如果提供了api_key，使用提供的key，否则从环境变量OPENAI_API_KEY读取
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = OpenAI()
            self.async_client = AsyncOpenAI()
    
    
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
        
        # OpenAI只支持frames模式
        if video_input_mode != "frames":
            if video_input_mode == "base64":
                warnings.warn(
                    "警告: OpenAI不支持视频base64模式，将使用'frames'模式（抽帧）。",
                    UserWarning
                )
            elif video_input_mode == "upload":
                warnings.warn(
                    "警告: OpenAI不支持'upload'模式，将使用'frames'模式（抽帧）。",
                    UserWarning
                )
            else:
                warnings.warn(
                    f"警告: 未知的视频输入模式'{video_input_mode}'，将使用'frames'模式。",
                    UserWarning
                )
            return "frames"
        
        return video_input_mode
    
    def _process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理消息列表，将本地图片路径转换为base64格式，将视频转换为多帧图片
        
        Args:
            messages: 原始消息列表
            
        Returns:
            处理后的消息列表
        """
        processed_messages = []
        
        for msg in messages:
            processed_msg = msg.copy()
            content = msg.get('content')
            
            # 如果content是列表（多模态消息）
            if isinstance(content, list):
                processed_content = []
                for item in content:
                    if isinstance(item, dict):
                        item_copy = item.copy()
                        
                        # 处理image_url
                        if item.get('type') == 'image_url':
                            image_url_data = item.get('image_url', {})
                            if isinstance(image_url_data, dict):
                                url = image_url_data.get('url', '')
                            else:
                                url = image_url_data
                            
                            # 如果是本地文件路径，转换为base64
                            if is_local_file(url):
                                base64_url = local_image_to_base64(url, include_mime=True)
                                if isinstance(image_url_data, dict):
                                    item_copy['image_url'] = {'url': base64_url}
                                else:
                                    item_copy['image_url'] = base64_url
                            
                            processed_content.append(item_copy)
                        
                        # 处理video_url：OpenAI不原生支持视频，需要抽帧转为多图
                        elif item.get('type') == 'video_url':
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
                                
                                # 将每一帧作为一个image_url添加到content中
                                for frame_base64 in frame_base64_list:
                                    processed_content.append({
                                        'type': 'image_url',
                                        'image_url': {'url': frame_base64}
                                    })
                            else:
                                # URL或base64格式的视频暂不支持，跳过
                                warnings.warn(
                                    f"OpenAI不支持视频URL或base64格式，仅支持本地视频文件抽帧。视频 {video_path} 将被跳过。",
                                    UserWarning
                                )
                        
                        else:
                            processed_content.append(item_copy)
                    else:
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
        OpenAI模型的completion调用
        
        Args:
            model: 模型名称 (如 gpt-5, gpt-5-mini, gpt-4.1)
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            reasoning_effort: 推理强度 ("low", "medium", "high")，仅gpt-5系列支持
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            return_full_response: 是否返回完整的response对象，默认False返回文本
            video_input_mode: 视频输入模式（OpenAI只支持"frames"模式）
            **kwargs: 其他OpenAI API参数
            
        Returns:
            - 当return_full_response=False时：返回字符串（输出内容）
            - 当return_full_response=True时：返回完整的API响应对象
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # 验证并标准化video_input_mode（OpenAI只支持frames）
        video_input_mode = self._validate_video_input_mode(video_input_mode)
        
        # 处理消息（将本地图片路径转换为base64）
        processed_messages = self._process_messages(messages)
        
        # 构建API调用参数
        api_params = {
            'model': model,
            'messages': processed_messages,
        }
        
        # 处理reasoning_effort参数
        if reasoning_effort is not None:
            if self._supports_reasoning(model):
                api_params['reasoning_effort'] = reasoning_effort
            else:
                warnings.warn(
                    f"警告: 模型 {model} 不支持 reasoning_effort 参数，该参数将被忽略。"
                    f"只有 {', '.join(self.REASONING_MODELS)} 系列模型支持该参数。",
                    UserWarning
                )
        
        # 添加其他可选参数
        if temperature is not None:
            api_params['temperature'] = temperature
        if max_tokens is not None:
            warnings.warn(
                "警告: OpenAI已不支持 max_tokens 参数，该参数将被忽略。"
                "请使用 max_completion_tokens 参数代替。",
                UserWarning
            )
        if top_p is not None:
            api_params['top_p'] = top_p
        
        # 添加其他额外参数
        api_params.update(kwargs)
        
        # 调用OpenAI API
        response = self.client.chat.completions.create(**api_params)
        
        # 根据参数决定返回格式
        if return_full_response:
            return response
        else:
            # 返回解析后的文本内容
            return response.choices[0].message.content
    
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
        OpenAI模型的异步completion调用
        
        Args:
            model: 模型名称 (如 gpt-5, gpt-5-mini, gpt-4.1)
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            reasoning_effort: 推理强度 ("low", "medium", "high")，仅gpt-5系列支持
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            return_full_response: 是否返回完整的response对象，默认False返回文本
            video_input_mode: 视频输入模式（OpenAI只支持"frames"模式）
            **kwargs: 其他OpenAI API参数
            
        Returns:
            - 当return_full_response=False时：返回字符串（输出内容）
            - 当return_full_response=True时：返回完整的API响应对象
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # 验证并标准化video_input_mode（OpenAI只支持frames）
        video_input_mode = self._validate_video_input_mode(video_input_mode)
        
        # 处理消息（将本地图片路径转换为base64）
        processed_messages = self._process_messages(messages)
        
        # 构建API调用参数
        api_params = {
            'model': model,
            'messages': processed_messages,
        }
        
        # 处理reasoning_effort参数
        if reasoning_effort is not None:
            if self._supports_reasoning(model):
                api_params['reasoning_effort'] = reasoning_effort
            else:
                warnings.warn(
                    f"警告: 模型 {model} 不支持 reasoning_effort 参数，该参数将被忽略。"
                    f"只有 {', '.join(self.REASONING_MODELS)} 系列模型支持该参数。",
                    UserWarning
                )
        
        # 添加其他可选参数
        if temperature is not None:
            api_params['temperature'] = temperature
        if max_tokens is not None:
            warnings.warn(
                "警告: OpenAI已不支持 max_tokens 参数，该参数将被忽略。"
                "请使用 max_completion_tokens 参数代替。",
                UserWarning
            )
        if top_p is not None:
            api_params['top_p'] = top_p
        
        # 添加其他额外参数
        api_params.update(kwargs)
        
        # 异步调用OpenAI API（唯一的改动）
        response = await self.async_client.chat.completions.create(**api_params)
        
        # 根据参数决定返回格式
        if return_full_response:
            return response
        else:
            # 返回解析后的文本内容
            return response.choices[0].message.content

