"""
SiliconFlow 模型调用实现
"""

import os
import warnings
from typing import Any, Dict, List, Optional

from openai import OpenAI, AsyncOpenAI

from .base import BaseLLMProvider
from .utils import (
    is_url, is_base64_data_url, is_local_file, 
    local_image_to_base64, local_video_to_base64,
    compress_video_if_needed, extract_video_frames_as_base64
)
import requests


class SiliconFlowProvider(BaseLLMProvider):
    """SiliconFlow模型Provider"""
    
    # 只支持思考模式的模型（必须使用流式）
    THINKING_ONLY_MODELS = [
        'Qwen/Qwen3-VL-235B-A22B-Thinking',
    ]
    
    # 只支持非思考模式的模型
    NON_THINKING_MODELS = [
        'Qwen/Qwen3-VL-235B-A22B-Instruct',
    ]
    
    # 既可以思考也可以不思考的模型（思考时必须使用流式）
    FLEXIBLE_MODELS = [
    ]
    
    # 所有支持的模型
    SUPPORTED_MODELS = THINKING_ONLY_MODELS + NON_THINKING_MODELS + FLEXIBLE_MODELS
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """
        初始化SiliconFlow Provider
        
        Args:
            api_key: SiliconFlow API密钥，如果不提供则从环境变量SILICONFLOW_API_KEY读取
            base_url: API基础URL，默认为SiliconFlow的API端点
            **kwargs: 其他配置参数
        """
        super().__init__(api_key, **kwargs)
        
        # 默认base_url
        if base_url is None:
            base_url = "https://api.siliconflow.cn/v1"
        
        # 如果提供了api_key，使用提供的key，否则从环境变量SILICONFLOW_API_KEY读取
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
            api_key_from_env = os.environ.get("SILICONFLOW_API_KEY")
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
    
    def _is_thinking_only_model(self, model: str) -> bool:
        """判断是否为只支持思考的模型"""
        return any(model.startswith(m) or model == m for m in self.THINKING_ONLY_MODELS)
    
    def _is_non_thinking_model(self, model: str) -> bool:
        """判断是否为只支持非思考的模型"""
        return any(model.startswith(m) or model == m for m in self.NON_THINKING_MODELS)
    
    def _is_flexible_model(self, model: str) -> bool:
        """判断是否为灵活模型（可思考可不思考）"""
        return any(model.startswith(m) or model == m for m in self.FLEXIBLE_MODELS)
    
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
                    "警告: SiliconFlow不支持'upload'模式，将使用默认的'base64'模式。",
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
    
    def _should_enable_thinking(self, model: str, reasoning_effort: Optional[str]) -> bool:
        """
        判断是否应该开启思考模式
        
        Args:
            model: 模型名称
            reasoning_effort: 推理强度参数
            
        Returns:
            是否开启思考
        """
        # 只支持思考的模型，总是开启
        if self._is_thinking_only_model(model):
            return True
        
        # 只支持非思考的模型，总是不开启
        if self._is_non_thinking_model(model):
            return False
        
        # 灵活模型，根据reasoning_effort判断
        if self._is_flexible_model(model):
            if reasoning_effort is None or reasoning_effort == '':
                return False  # 默认不开启
            reasoning_effort_lower = reasoning_effort.lower()
            # minimal或none表示不思考
            if reasoning_effort_lower in ['minimal', 'none']:
                return False
            else:
                return True
        
        return False
    
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
                            
                            # 如果是本地文件路径，转换为base64
                            if is_local_file(url):
                                base64_url = local_image_to_base64(url, include_mime=True)
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
                                # base64模式：视频直接转base64（SiliconFlow限制10MB）
                                if is_local_file(url):
                                    # 检查并压缩视频
                                    # base64编码会增加约33%大小，所以压缩到7.5MB以确保编码后不超过10MB
                                    processed_path = compress_video_if_needed(url, max_size_mb=7.5)
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
                                    
                                    # base64编码会增加约33%大小，所以限制为7.5MB
                                    if video_size_mb > 7.5:
                                        # 需要压缩，先保存为临时文件
                                        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
                                        import os
                                        os.close(temp_fd)
                                        
                                        with open(temp_path, 'wb') as f:
                                            f.write(video_data)
                                        
                                        temp_files.append(temp_path)
                                        processed_path = compress_video_if_needed(temp_path, max_size_mb=7.5)
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
        
        # 存储临时文件列表，用于后续清理
        if temp_files:
            processed_messages._temp_files = temp_files
        
        return processed_messages
    
    def _collect_stream_response(self, stream_response: Any) -> tuple:
        """
        收集流式响应的内容
        
        Args:
            stream_response: 流式响应对象
            
        Returns:
            (reasoning_content, answer_content) 元组
        """
        reasoning_content = ""
        answer_content = ""
        
        for chunk in stream_response:
            if not chunk.choices:
                # 包含usage信息的chunk，跳过
                continue
            
            delta = chunk.choices[0].delta
            
            # 收集思考内容
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content
            
            # 收集回复内容
            if hasattr(delta, 'content') and delta.content is not None:
                answer_content += delta.content
        
        return reasoning_content, answer_content
    
    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        return_full_response: bool = False,
        video_input_mode: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        SiliconFlow模型的completion调用
        
        Args:
            model: 模型名称
            messages: 消息列表
            reasoning_effort: 推理强度，用于控制是否开启思考
                - None/minimal/none: 不思考（对于灵活模型）
                - low/medium/high: 开启思考（对于灵活模型）
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            thinking_budget: 思考过程的最大Token数，默认8192
            return_full_response: 是否返回完整的response对象
            video_input_mode: 视频输入模式（"base64"或"frames"，默认"base64"）
            **kwargs: 其他参数
            
        Returns:
            - 当return_full_response=False时：
                - 如果没有思考内容：返回字符串（输出内容）
                - 如果有思考内容：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回响应对象（思考模式下为模拟对象）
            
        Note:
            - 思考模式必须使用流式API
            - thinking_only模型总是开启思考
            - non_thinking模型总是不开启思考
            - flexible模型根据reasoning_effort决定
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # 判断是否开启思考
        enable_thinking = self._should_enable_thinking(model, reasoning_effort)
        
        # 如果用户对不支持思考的模型传入了reasoning_effort，发出警告
        if reasoning_effort and self._is_non_thinking_model(model):
            warnings.warn(
                f"警告: 模型 {model} 不支持思考功能，reasoning_effort参数将被忽略。",
                UserWarning
            )
        
        # 验证并标准化video_input_mode
        video_input_mode = self._validate_video_input_mode(video_input_mode)
        
        # 处理消息（将本地图片路径转换为base64）
        processed_messages = self._process_messages(messages, video_input_mode)
        
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
        
        # 处理思考模式
        if enable_thinking:
            # 思考模式必须使用流式
            api_params['stream'] = True
            api_params['stream_options'] = {'include_usage': True}
            
            # 设置extra_body参数
            # extra_body = {'enable_thinking': True}
            extra_body = {}
            
            # 设置thinking_budget
            if thinking_budget is not None:
                extra_body['thinking_budget'] = thinking_budget
            else:
                extra_body['thinking_budget'] = 8192  # 默认值
            
            api_params['extra_body'] = extra_body
            
            # 调用流式API
            stream_response = self.client.chat.completions.create(**api_params)
            
            # 收集流式响应
            reasoning_content, answer_content = self._collect_stream_response(stream_response)
            
            # 根据return_full_response决定返回格式
            if return_full_response:
                # 创建一个模拟的response对象
                class MockResponse:
                    def __init__(self, content, reasoning_content):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': content,
                                'reasoning_content': reasoning_content,
                                'role': 'assistant'
                            })()
                        })()]
                        self.thinking_content = reasoning_content
                
                return MockResponse(answer_content, reasoning_content)
            else:
                # 返回解析后的内容
                if reasoning_content:
                    return {
                        "content": answer_content,
                        "thinking_content": reasoning_content
                    }
                else:
                    return answer_content
        
        else:
            # 非思考模式，使用普通API
            # 添加其他额外参数
            api_params.update(kwargs)
            
            # 调用API
            response = self.client.chat.completions.create(**api_params)
            
            # 根据return_full_response决定返回格式
            if return_full_response:
                return response
            else:
                return response.choices[0].message.content
    
    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        thinking_budget: Optional[int] = None,
        return_full_response: bool = False,
        video_input_mode: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        SiliconFlow模型的异步completion调用
        
        Args:
            model: 模型名称
            messages: 消息列表
            reasoning_effort: 推理强度，用于控制是否开启思考
                - None/minimal/none: 不思考（对于灵活模型）
                - low/medium/high: 开启思考（对于灵活模型）
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            thinking_budget: 思考过程的最大Token数，默认8192
            return_full_response: 是否返回完整的response对象
            video_input_mode: 视频输入模式（"base64"或"frames"，默认"base64"）
            **kwargs: 其他参数
            
        Returns:
            - 当return_full_response=False时：
                - 如果没有思考内容：返回字符串（输出内容）
                - 如果有思考内容：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回响应对象（思考模式下为模拟对象）
            
        Note:
            - 思考模式必须使用流式API
            - thinking_only模型总是开启思考
            - non_thinking模型总是不开启思考
            - flexible模型根据reasoning_effort决定
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # 判断是否开启思考
        enable_thinking = self._should_enable_thinking(model, reasoning_effort)
        
        # 如果用户对不支持思考的模型传入了reasoning_effort，发出警告
        if reasoning_effort and self._is_non_thinking_model(model):
            warnings.warn(
                f"警告: 模型 {model} 不支持思考功能，reasoning_effort参数将被忽略。",
                UserWarning
            )
        
        # 验证并标准化video_input_mode
        video_input_mode = self._validate_video_input_mode(video_input_mode)
        
        # 处理消息（将本地图片路径转换为base64）
        processed_messages = self._process_messages(messages, video_input_mode)
        
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
        
        # 处理思考模式
        if enable_thinking:
            # 思考模式必须使用流式
            api_params['stream'] = True
            api_params['stream_options'] = {'include_usage': True}
            
            # 设置extra_body参数
            # extra_body = {'enable_thinking': True}
            extra_body = {}
            
            # 设置thinking_budget
            if thinking_budget is not None:
                extra_body['thinking_budget'] = thinking_budget
            
            api_params['extra_body'] = extra_body
            
            # 异步调用流式API
            stream_response = await self.async_client.chat.completions.create(**api_params)
            
            # 收集流式响应（注意：这里需要异步迭代）
            reasoning_content = ""
            answer_content = ""
            
            async for chunk in stream_response:
                if not hasattr(chunk, 'choices') or len(chunk.choices) == 0:
                    continue
                
                delta = chunk.choices[0].delta
                
                # 收集思考内容
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                
                # 收集回复内容
                if hasattr(delta, 'content') and delta.content is not None:
                    answer_content += delta.content
            
            # 根据return_full_response决定返回格式
            if return_full_response:
                # 创建一个模拟的response对象
                class MockResponse:
                    def __init__(self, content, reasoning_content):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': content,
                                'reasoning_content': reasoning_content,
                                'role': 'assistant'
                            })()
                        })()]
                        self.thinking_content = reasoning_content
                
                return MockResponse(answer_content, reasoning_content)
            else:
                # 返回解析后的内容
                if reasoning_content:
                    return {
                        "content": answer_content,
                        "thinking_content": reasoning_content
                    }
                else:
                    return answer_content
        
        else:
            # 非思考模式，使用普通API
            # 添加其他额外参数
            api_params.update(kwargs)
            
            # 异步调用API
            response = await self.async_client.chat.completions.create(**api_params)
            
            # 根据return_full_response决定返回格式
            if return_full_response:
                return response
            else:
                return response.choices[0].message.content
