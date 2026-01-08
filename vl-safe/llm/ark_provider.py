"""
Ark (火山引擎豆包) 模型调用实现
"""

import os
import warnings
from typing import Any, Dict, List, Optional

from openai import OpenAI, AsyncOpenAI

from .base import BaseLLMProvider
from .utils import (
    is_url, is_base64_data_url, is_local_file, 
    local_image_to_base64, local_video_to_base64,
    compress_video_if_needed, compress_image_if_needed,
    extract_video_frames_as_base64
)
import requests


class ArkProvider(BaseLLMProvider):
    """Ark (火山引擎豆包) 模型Provider"""
    
    # 支持的模型列表
    SUPPORTED_MODELS = [
        'doubao-seed-1-6-251015',
        'doubao-seed-1-6-vision-250815',
    ]
    
    # 支持reasoning_effort参数的模型（使用标准reasoning_effort参数）
    REASONING_EFFORT_MODELS = [
        'doubao-seed-1-6-251015',
    ]
    
    # 使用extra_body thinking参数的模型
    EXTRA_BODY_THINKING_MODELS = [
        'doubao-seed-1-6-vision-250815',
    ]
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """
        初始化Ark Provider
        
        Args:
            api_key: Ark API密钥，如果不提供则从环境变量ARK_API_KEY读取
            base_url: API基础URL，默认为火山引擎北京区域
            **kwargs: 其他配置参数
        """
        super().__init__(api_key, **kwargs)
        
        # 默认base_url
        if base_url is None:
            base_url = "https://ark.cn-beijing.volces.com/api/v3"
        
        # 如果提供了api_key，使用提供的key，否则从环境变量ARK_API_KEY读取
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=kwargs.get('timeout', 1800)  # 深度思考建议设置更长的超时时间
            )
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=kwargs.get('timeout', 1800)
            )
        else:
            # 从环境变量读取
            api_key_from_env = os.environ.get("ARK_API_KEY")
            self.client = OpenAI(
                api_key=api_key_from_env,
                base_url=base_url,
                timeout=kwargs.get('timeout', 1800)
            )
            self.async_client = AsyncOpenAI(
                api_key=api_key_from_env,
                base_url=base_url,
                timeout=kwargs.get('timeout', 1800)
            )
    
    def _supports_reasoning_effort(self, model: str) -> bool:
        """
        判断模型是否支持reasoning_effort参数
        
        Args:
            model: 模型名称
            
        Returns:
            是否支持reasoning_effort
        """
        return any(model.startswith(m) or model == m 
                   for m in self.REASONING_EFFORT_MODELS)
    
    def _uses_extra_body_thinking(self, model: str) -> bool:
        """
        判断模型是否使用extra_body的thinking参数
        
        Args:
            model: 模型名称
            
        Returns:
            是否使用extra_body thinking
        """
        return any(model.startswith(m) or model == m 
                   for m in self.EXTRA_BODY_THINKING_MODELS)
    
    def _convert_reasoning_effort(self, reasoning_effort: str) -> str:
        """
        转换OpenAI风格的reasoning_effort到Ark风格
        OpenAI: None表示不思考
        Ark: minimal表示不思考，支持 minimal, low, medium, high
        
        Args:
            reasoning_effort: OpenAI风格的reasoning_effort
            
        Returns:
            Ark风格的reasoning_effort
        """
        # OpenAI的None或者空字符串转为Ark的minimal
        if reasoning_effort is None or reasoning_effort == '' or reasoning_effort.lower() == 'none':
            return 'minimal'
        return reasoning_effort
    
    def _convert_to_thinking_type(self, reasoning_effort: Optional[str]) -> str:
        """
        转换reasoning_effort到extra_body的thinking.type
        
        Args:
            reasoning_effort: reasoning_effort值
            
        Returns:
            thinking.type值 (disabled, enabled, auto)
        """
        if reasoning_effort is None or reasoning_effort == '':
            return 'auto'  # 默认为auto
        
        reasoning_effort_lower = reasoning_effort.lower()
        
        # minimal或none转为disabled
        if reasoning_effort_lower in ['minimal', 'none']:
            return 'disabled'
        # low, medium, high转为enabled
        elif reasoning_effort_lower in ['low', 'medium', 'high']:
            return 'enabled'
        else:
            return 'auto'
    
    def _validate_video_input_mode(self, video_input_mode: Optional[str]) -> str:
        """
        验证并标准化视频输入模式
        
        Args:
            video_input_mode: 视频输入模式
            
        Returns:
            标准化的视频输入模式
        """
        if video_input_mode is None:
            return "base64"  # 默认使用base64模式
        
        video_input_mode = video_input_mode.lower()
        
        # 验证模式是否支持
        if video_input_mode not in ["base64", "frames"]:
            if video_input_mode == "upload":
                warnings.warn(
                    "警告: Ark不支持'upload'模式，将使用默认的'base64'模式。",
                    UserWarning
                )
                return "base64"
            else:
                warnings.warn(
                    f"警告: 未知的视频输入模式'{video_input_mode}'，将使用'base64'模式。",
                    UserWarning
                )
                return "base64"
        
        return video_input_mode
    
    def _process_messages(self, messages: List[Dict[str, Any]], video_input_mode: str = "base64") -> List[Dict[str, Any]]:
        """
        处理消息列表，将本地图片和视频路径转换为base64格式或抽帧
        
        Args:
            messages: 原始消息列表
            video_input_mode: 视频输入模式（"base64"或"frames"）
            
        Returns:
            处理后的消息列表
        """
        import tempfile
        
        processed_messages = []
        temp_files = []  # 记录临时文件，用于后续清理
        
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
                            
                            # 如果是本地文件路径，先压缩再转换为base64
                            if is_local_file(url):
                                # Ark对图片大小有限制（10MB），需要提前压缩
                                processed_path = compress_image_if_needed(url, max_size_mb=7.5)
                                if processed_path != url:
                                    temp_files.append(processed_path)
                                
                                base64_url = local_image_to_base64(processed_path, include_mime=True)
                                if isinstance(image_url_data, dict):
                                    item_copy['image_url'] = {'url': base64_url}
                                else:
                                    item_copy['image_url'] = base64_url
                        
                        # 处理video_url: 支持base64和frames模式
                        elif item.get('type') == 'video_url':
                            video_url_data = item.get('video_url') or item.get('video')
                            if isinstance(video_url_data, dict):
                                url = video_url_data.get('url', '')
                            else:
                                url = video_url_data
                            
                            # 移除 file:// 前缀
                            if url.startswith('file://'):
                                url = url[7:]
                            
                            # Base64格式不处理
                            if is_base64_data_url(url):
                                processed_content.append(item_copy)
                                continue
                            
                            # 根据模式处理视频
                            if video_input_mode == "frames":
                                # 抽帧模式：将视频转换为多张图片
                                if is_local_file(url):
                                    # 获取抽帧参数
                                    fps = item.get('fps', None)
                                    max_frames = item.get('max_frames', None)
                                    
                                    # 如果都没提供，默认fps=1
                                    if fps is None and max_frames is None:
                                        fps = 1.0
                                    
                                    # 抽取视频帧并转换为base64
                                    frame_base64_list = extract_video_frames_as_base64(
                                        url, 
                                        fps=fps, 
                                        max_frames=max_frames
                                    )
                                    
                                    # 将每一帧作为一个image_url添加到content中
                                    for frame_base64 in frame_base64_list:
                                        processed_content.append({
                                            'type': 'image_url',
                                            'image_url': {'url': frame_base64}
                                        })
                                    
                                    # 跳过添加原video_url
                                    continue
                                
                                elif is_url(url):
                                    # URL视频：下载后抽帧
                                    response = requests.get(url)
                                    video_data = response.content
                                    
                                    # 保存为临时文件
                                    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
                                    import os
                                    os.close(temp_fd)
                                    
                                    with open(temp_path, 'wb') as f:
                                        f.write(video_data)
                                    
                                    temp_files.append(temp_path)
                                    
                                    # 获取抽帧参数
                                    fps = item.get('fps', None)
                                    max_frames = item.get('max_frames', None)
                                    if fps is None and max_frames is None:
                                        fps = 2.0
                                    
                                    # 抽帧
                                    frame_base64_list = extract_video_frames_as_base64(
                                        temp_path, 
                                        fps=fps, 
                                        max_frames=max_frames
                                    )
                                    
                                    # 添加帧到content
                                    for frame_base64 in frame_base64_list:
                                        processed_content.append({
                                            'type': 'image_url',
                                            'image_url': {'url': frame_base64}
                                        })
                                    
                                    continue
                            
                            else:
                                # base64模式：视频直接转base64（Ark限制50MB）
                                if is_local_file(url):
                                    # 检查并压缩视频
                                    processed_path = compress_video_if_needed(url, max_size_mb=50.0)
                                    if processed_path != url:
                                        temp_files.append(processed_path)
                                    
                                    # 转换为base64
                                    base64_url = local_video_to_base64(processed_path, include_mime=True)
                                    if isinstance(video_url_data, dict):
                                        item_copy['video_url'] = {'url': base64_url}
                                    else:
                                        item_copy['video_url'] = base64_url
                                
                                elif is_url(url):
                                    # 下载视频
                                    response = requests.get(url)
                                    video_data = response.content
                                    video_size_mb = len(video_data) / (1024 * 1024)
                                    
                                    if video_size_mb > 50.0:
                                        # 需要压缩，先保存为临时文件
                                        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
                                        import os
                                        os.close(temp_fd)
                                        
                                        with open(temp_path, 'wb') as f:
                                            f.write(video_data)
                                        
                                        temp_files.append(temp_path)
                                        processed_path = compress_video_if_needed(temp_path, max_size_mb=50.0)
                                        if processed_path != temp_path:
                                            temp_files.append(processed_path)
                                        
                                        base64_url = local_video_to_base64(processed_path, include_mime=True)
                                    else:
                                        # 直接转base64
                                        import base64
                                        from .utils import get_mime_type
                                        mime_type = get_mime_type(url, 'video_url')
                                        video_base64 = base64.b64encode(video_data).decode('utf-8')
                                        base64_url = f"data:{mime_type};base64,{video_base64}"
                                    
                                    if isinstance(video_url_data, dict):
                                        item_copy['video_url'] = {'url': base64_url}
                                    else:
                                        item_copy['video_url'] = base64_url
                        
                        processed_content.append(item_copy)
                    else:
                        processed_content.append(item)
                
                processed_msg['content'] = processed_content
            
            processed_messages.append(processed_msg)
        
        # 注意：temp_files 中的临时文件会在进程结束时自动清理
        # 如果需要更早清理，可以在调用完成后手动删除
        
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
        Ark模型的completion调用
        
        Args:
            model: 模型名称
            messages: 消息列表
            reasoning_effort: 推理强度，会根据模型类型自动转换
                - 对于reasoning_effort模型: minimal, low, medium, high
                - 对于extra_body模型: minimal->disabled, low/medium/high->enabled
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            return_full_response: 是否返回完整的response对象，默认False返回解析后的内容
            video_input_mode: 视频输入模式（"base64"或"frames"，默认"base64"）
            **kwargs: 其他参数
            
        Returns:
            - 当return_full_response=False时：
                - 如果没有思考内容：返回字符串（输出内容）
                - 如果有思考内容：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回增强的响应对象（包含thinking_content属性）
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # 验证并标准化video_input_mode
        video_input_mode = self._validate_video_input_mode(video_input_mode)
        
        # 处理消息（将本地图片路径转换为base64）
        processed_messages = self._process_messages(messages, video_input_mode)
        
        # 构建API调用参数
        api_params = {
            'model': model,
            'messages': processed_messages,
        }
        
        # 根据模型类型处理reasoning_effort参数
        if self._supports_reasoning_effort(model):
            # 使用reasoning_effort参数的模型
            if reasoning_effort is not None:
                converted_effort = self._convert_reasoning_effort(reasoning_effort)
                api_params['reasoning_effort'] = converted_effort
        
        elif self._uses_extra_body_thinking(model):
            # 使用extra_body thinking参数的模型
            if reasoning_effort is not None:
                thinking_type = self._convert_to_thinking_type(reasoning_effort)
                api_params['extra_body'] = {
                    'thinking': {
                        'type': thinking_type
                    }
                }
            else:
                # 默认使用auto
                api_params['extra_body'] = {
                    'thinking': {
                        'type': 'enabled'
                    }
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
        
        # 调用Ark API
        response = self.client.chat.completions.create(**api_params)
        
        # 增强返回对象，添加thinking_content访问方法
        enhanced_response = self._enhance_response(response)
        
        # 根据参数决定返回格式
        if return_full_response:
            return enhanced_response
        else:
            # 返回解析后的内容
            content = enhanced_response.choices[0].message.content
            thinking_content = enhanced_response.thinking_content
            
            # 如果有思考内容，返回字典；否则只返回文本
            if thinking_content:
                return {
                    "content": content,
                    "thinking_content": thinking_content
                }
            else:
                return content
    
    def _enhance_response(self, response: Any) -> Any:
        """
        增强响应对象，方便访问thinking_content
        
        Args:
            response: 原始API响应
            
        Returns:
            增强后的响应对象
        """
        # Ark的思考内容在message.reasoning_content中
        # 为了方便使用，我们添加一个helper方法
        if hasattr(response, 'choices') and len(response.choices) > 0:
            message = response.choices[0].message
            
            # 如果没有thinking_content属性，添加一个方便访问的属性
            if not hasattr(response, 'thinking_content'):
                if hasattr(message, 'reasoning_content'):
                    response.thinking_content = message.reasoning_content
                else:
                    response.thinking_content = None
        
        return response
    
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
        Ark模型的异步completion调用
        
        Args:
            model: 模型名称
            messages: 消息列表
            reasoning_effort: 推理强度，会根据模型类型自动转换
                - 对于reasoning_effort模型: minimal, low, medium, high
                - 对于extra_body模型: minimal->disabled, low/medium/high->enabled
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            return_full_response: 是否返回完整的response对象，默认False返回解析后的内容
            video_input_mode: 视频输入模式（"base64"或"frames"，默认"base64"）
            **kwargs: 其他参数
            
        Returns:
            - 当return_full_response=False时：
                - 如果没有思考内容：返回字符串（输出内容）
                - 如果有思考内容：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回增强的响应对象（包含thinking_content属性）
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # 验证并标准化video_input_mode
        video_input_mode = self._validate_video_input_mode(video_input_mode)
        
        # 处理消息（将本地图片路径转换为base64）
        processed_messages = self._process_messages(messages, video_input_mode)
        
        # 构建API调用参数
        api_params = {
            'model': model,
            'messages': processed_messages,
        }
        
        # 根据模型类型处理reasoning_effort参数
        if self._supports_reasoning_effort(model):
            # 使用reasoning_effort参数的模型
            if reasoning_effort is not None:
                converted_effort = self._convert_reasoning_effort(reasoning_effort)
                api_params['reasoning_effort'] = converted_effort
        
        elif self._uses_extra_body_thinking(model):
            # 使用extra_body thinking参数的模型
            if reasoning_effort is not None:
                thinking_type = self._convert_to_thinking_type(reasoning_effort)
                api_params['extra_body'] = {
                    'thinking': {
                        'type': thinking_type
                    }
                }
            else:
                # 默认使用auto
                api_params['extra_body'] = {
                    'thinking': {
                        'type': 'enabled'
                    }
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
        
        # 异步调用Ark API
        response = await self.async_client.chat.completions.create(**api_params)
        
        # 增强返回对象，添加thinking_content访问方法
        enhanced_response = self._enhance_response(response)
        
        # 根据参数决定返回格式
        if return_full_response:
            return enhanced_response
        else:
            # 返回解析后的内容
            content = enhanced_response.choices[0].message.content
            thinking_content = enhanced_response.thinking_content
            
            # 如果有思考内容，返回字典；否则只返回文本
            if thinking_content:
                return {
                    "content": content,
                    "thinking_content": thinking_content
                }
            else:
                return content

