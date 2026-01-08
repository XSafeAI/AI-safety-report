"""
Gemini (Google GenAI) 模型调用实现
"""

import os
import warnings
import base64
import requests
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types

from .base import BaseLLMProvider
from .utils import is_url, is_base64_data_url, is_local_file, get_mime_type


class GeminiProvider(BaseLLMProvider):
    """Gemini模型Provider"""
    
    # Gemini 3 系列模型（使用thinkingLevel参数）
    GEMINI_3_MODELS = [
        'gemini-3-pro-preview',
    ]
    
    # Gemini 2.5 系列模型（使用thinkingBudget参数）
    GEMINI_2_5_MODELS = [
        'gemini-2.5-pro',
        'gemini-2.5-flash'
    ]
    
    # 所有支持的模型
    SUPPORTED_MODELS = GEMINI_3_MODELS + GEMINI_2_5_MODELS
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        初始化Gemini Provider
        
        Args:
            api_key: Gemini API密钥，如果不提供则从环境变量GEMINI_API_KEY读取
            **kwargs: 其他配置参数
        """
        super().__init__(api_key, **kwargs)
        
        # 创建客户端
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            # 从环境变量读取
            api_key_from_env = os.environ.get("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key_from_env)
    
    def _is_gemini_3_model(self, model: str) -> bool:
        """判断是否为Gemini 3系列模型"""
        return any(model.startswith(m) or model == m for m in self.GEMINI_3_MODELS)
    
    def _is_gemini_2_5_model(self, model: str) -> bool:
        """判断是否为Gemini 2.5系列模型"""
        return any(model.startswith(m) or model == m for m in self.GEMINI_2_5_MODELS)
    
    def _validate_video_input_mode(self, video_input_mode: Optional[str]) -> str:
        """
        验证并标准化视频输入模式
        
        Args:
            video_input_mode: 视频输入模式
            
        Returns:
            标准化的视频输入模式
        """
        if video_input_mode is None:
            return "upload"  # 默认使用upload模式
        
        video_input_mode = video_input_mode.lower()
        
        # Gemini只支持upload模式
        if video_input_mode != "upload":
            if video_input_mode == "base64":
                warnings.warn(
                    "警告: Gemini不推荐使用base64模式传输视频（大文件会自动上传），将使用'upload'模式。",
                    UserWarning
                )
            elif video_input_mode == "frames":
                warnings.warn(
                    "警告: Gemini不支持'frames'模式，将使用'upload'模式。",
                    UserWarning
                )
            else:
                warnings.warn(
                    f"警告: 未知的视频输入模式'{video_input_mode}'，将使用'upload'模式。",
                    UserWarning
                )
            return "upload"
        
        return video_input_mode
    
    def _convert_reasoning_effort_to_thinking_level(self, reasoning_effort: Optional[str]) -> Optional[str]:
        """
        转换reasoning_effort到Gemini 3的thinkingLevel参数
        
        Args:
            reasoning_effort: 推理强度
            
        Returns:
            thinkingLevel值 ("low", "high") 或 None
        """
        if reasoning_effort is None or reasoning_effort == '':
            return None  # 使用默认值
        
        reasoning_effort_lower = reasoning_effort.lower()
        
        if reasoning_effort_lower in ['none', 'minimal']:
            # Gemini 3 Pro无法停用思考，返回"low"
            warnings.warn(
                "注意: Gemini 3 Pro无法完全停用思考功能，将使用'low'级别。",
                UserWarning
            )
            return "low"
        elif reasoning_effort_lower == 'low':
            return "low"
        elif reasoning_effort_lower == 'medium':
            # medium映射到low，并提示
            warnings.warn(
                "注意: Gemini 3 Pro不支持'medium'级别，已转换为'low'。",
                UserWarning
            )
            return "low"
        elif reasoning_effort_lower == 'high':
            return "high"
        else:
            return None
    
    def _convert_reasoning_effort_to_thinking_budget(self, reasoning_effort: Optional[str], model: str) -> Optional[int]:
        """
        转换reasoning_effort到Gemini 2.5的thinkingBudget参数
        
        Args:
            reasoning_effort: 推理强度
            model: 模型名称
            
        Returns:
            thinkingBudget值
        """
        if reasoning_effort is None or reasoning_effort == '':
            return -1  # 默认为动态思考
        
        reasoning_effort_lower = reasoning_effort.lower()
        is_pro_model = 'pro' in model.lower() and 'flash' not in model.lower()
        
        if reasoning_effort_lower in ['none', 'minimal']:
            # 2.5 Pro无法完全停用思考，使用最小值128
            # Flash系列可以停用，使用0
            if is_pro_model:
                warnings.warn(
                    "注意: Gemini 2.5 Pro无法完全停用思考功能，将使用最小值thinkingBudget=128。",
                    UserWarning
                )
                return 128
            else:
                return 0  # Flash系列可以停用
        elif reasoning_effort_lower == 'low':
            # 低强度思考，使用较小的budget
            if 'flash' in model.lower():
                return 4096  # Flash系列使用较小值
            else:
                return 2048  # Pro系列使用较小值
        elif reasoning_effort_lower == 'medium':
            # 中等强度思考
            warnings.warn(
                "注意: 建议使用'low'或'high'，'medium'将被映射到中等预算值。",
                UserWarning
            )
            if 'flash' in model.lower():
                return 12288  # Flash系列
            else:
                return 8192  # Pro系列
        elif reasoning_effort_lower == 'high':
            # 高强度思考，使用较大的budget
            if 'flash' in model.lower():
                return 24576  # Flash系列最大值
            else:
                return 32768  # Pro系列最大值
        else:
            return -1  # 默认为动态思考
    
    
    def _upload_and_wait(self, file_path: str):
        """
        上传文件到Gemini并等待处理完成
        
        Args:
            file_path: 文件路径
            
        Returns:
            上传后的文件对象
        """
        uploaded_file = self.client.files.upload(file=file_path)
        
        # 等待文件处理完成（视频文件需要时间处理）
        file_name = uploaded_file.name
        max_wait_time = 300  # 最多等待5分钟
        wait_interval = 2  # 每2秒检查一次
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            file_info = self.client.files.get(name=file_name)
            if file_info.state == "ACTIVE":
                break
            elif file_info.state == "FAILED":
                raise Exception(f"文件处理失败: {file_name}")
            
            # 文件还在处理中，等待
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        if elapsed_time >= max_wait_time:
            warnings.warn(f"文件处理超时 ({max_wait_time}秒): {file_name}，将继续尝试使用", UserWarning)
        
        return uploaded_file
    
    def _process_media_content(self, content_item: Dict) -> Union[types.Part, Any]:
        """
        处理多媒体内容（图片、视频等）
        
        Args:
            content_item: 内容项，包含type和url/data
            
        Returns:
            处理后的Part对象或上传的文件对象
        """
        content_type = content_item.get('type')
        
        if content_type == 'text':
            return types.Part(text=content_item['text'])
        
        elif content_type in ['image_url', 'video_url']:
            url_data = content_item.get('image_url') or content_item.get('video_url')
            url = url_data.get('url') if isinstance(url_data, dict) else url_data
            
            # 判断是URL、本地路径还是Base64
            if is_url(url):
                # URL: 下载后转为bytes
                response = requests.get(url)
                data_bytes = response.content
                mime_type = get_mime_type(url, content_type)
                file_size = len(data_bytes)
                
                # 小文件（<20MB）：直接使用
                if file_size < 20 * 1024 * 1024:
                    if content_type == 'video_url':
                        # 视频使用inline_data
                        return types.Part(
                            inline_data=types.Blob(data=data_bytes, mime_type=mime_type)
                        )
                    else:
                        # 图片使用from_bytes
                        return types.Part.from_bytes(data=data_bytes, mime_type=mime_type)
                
                # 大文件（≥20MB）：保存为临时文件并上传
                else:
                    suffix = Path(url).suffix or ('.mp4' if 'video' in mime_type else '.jpg')
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(data_bytes)
                        tmp_path = tmp_file.name
                    
                    try:
                        return self._upload_and_wait(tmp_path)
                    finally:
                        # 清理临时文件
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
            
            elif url.startswith('data:'):
                # Base64 data URL: data:image/jpeg;base64,/9j/4AAQ...
                header, encoded = url.split(',', 1)
                mime_type = header.split(';')[0].split(':')[1]
                data_bytes = base64.b64decode(encoded)
                file_size = len(data_bytes)
                
                # 小文件（<20MB）：直接使用
                if file_size < 20 * 1024 * 1024:
                    if 'video' in mime_type:
                        # 视频使用inline_data
                        return types.Part(
                            inline_data=types.Blob(data=data_bytes, mime_type=mime_type)
                        )
                    else:
                        # 图片使用from_bytes
                        return types.Part.from_bytes(data=data_bytes, mime_type=mime_type)
                
                # 大文件（≥20MB）：保存为临时文件并上传
                else:
                    # 根据mime_type确定文件后缀
                    if 'video' in mime_type:
                        suffix = '.mp4'
                    elif 'image' in mime_type:
                        suffix = '.jpg'
                    else:
                        suffix = '.dat'
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(data_bytes)
                        tmp_path = tmp_file.name
                    
                    try:
                        return self._upload_and_wait(tmp_path)
                    finally:
                        # 清理临时文件
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
            
            elif is_local_file(url):
                # 本地文件路径：检查文件大小
                file_path = Path(url)
                file_size = file_path.stat().st_size
                mime_type = get_mime_type(url, content_type)
                
                # 如果是视频且小于20MB，直接读取为Base64使用inline_data
                if content_type == 'video_url' and file_size < 20 * 1024 * 1024:
                    with open(file_path, 'rb') as f:
                        data_bytes = f.read()
                    return types.Part(
                        inline_data=types.Blob(data=data_bytes, mime_type=mime_type)
                    )
                
                # 如果是图片，直接使用from_bytes（自动处理）
                if content_type == 'image_url':
                    with open(file_path, 'rb') as f:
                        data_bytes = f.read()
                    return types.Part.from_bytes(data=data_bytes, mime_type=mime_type)
                
                # 大视频（>=20MB）：上传文件到Gemini服务器
                return self._upload_and_wait(url)
            
            else:
                # 尝试作为Base64字符串处理
                try:
                    data_bytes = base64.b64decode(url)
                    mime_type = get_mime_type('', content_type)
                    return types.Part.from_bytes(data=data_bytes, mime_type=mime_type)
                except Exception as e:
                    warnings.warn(f"无法处理媒体内容: {e}", UserWarning)
                    return None
        
        return None
    
    def _convert_messages_to_contents(self, messages: List[Dict[str, Any]]) -> Any:
        """
        转换消息格式为Gemini的contents格式
        
        Args:
            messages: OpenAI风格的消息列表
            
        Returns:
            Gemini风格的contents
        """
        # 如果只有一条简单的user消息，直接返回
        if len(messages) == 1 and messages[0].get('role') == 'user':
            content = messages[0]['content']
            
            # 如果是字符串，直接返回
            if isinstance(content, str):
                return content
            
            # 如果是列表（多模态），处理每个部分
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        part = self._process_media_content(item)
                        if part is not None:
                            parts.append(part)
                    elif isinstance(item, str):
                        parts.append(item)
                
                return parts if parts else content
        
        # 多轮对话的情况：转换为Gemini的Content格式
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # 处理system消息：Gemini使用system_instruction
            if role == 'system':
                # 将system消息存储，稍后会在API调用时使用
                # 但目前的generate_content调用不支持system_instruction参数
                # 所以我们将system消息转换为user消息的前缀
                if system_instruction is None:
                    system_instruction = content
                else:
                    system_instruction += "\n\n" + content
                continue
            
            # 处理user和assistant消息
            if role in ['user', 'assistant']:
                # Gemini的role名称：user -> user, assistant -> model
                gemini_role = 'model' if role == 'assistant' else 'user'
                
                # 处理content内容
                parts = []
                
                if isinstance(content, str):
                    # 简单文本消息
                    # 如果是第一个user消息且有system_instruction，添加到前面
                    if gemini_role == 'user' and system_instruction and not contents:
                        parts.append(types.Part(text=f"{system_instruction}\n\n{content}"))
                        system_instruction = None  # 只添加一次
                    else:
                        parts.append(types.Part(text=content))
                
                elif isinstance(content, list):
                    # 多模态消息
                    # 如果是第一个user消息且有system_instruction，先添加system指令
                    if gemini_role == 'user' and system_instruction and not contents:
                        parts.append(types.Part(text=system_instruction))
                        system_instruction = None  # 只添加一次
                    
                    # 处理多模态内容
                    for item in content:
                        if isinstance(item, dict):
                            part = self._process_media_content(item)
                            if part is not None:
                                parts.append(part)
                        elif isinstance(item, str):
                            parts.append(types.Part(text=item))
                
                # 创建Content对象
                if parts:
                    contents.append(types.Content(
                        role=gemini_role,
                        parts=parts
                    ))
        
        # 如果处理后为空，返回空字符串
        if not contents:
            return ""
        
        return contents
    
    def _should_include_thoughts(self, reasoning_effort: Optional[str]) -> bool:
        """
        判断是否应该包含思考总结
        
        Args:
            reasoning_effort: 推理强度
            
        Returns:
            是否包含思考总结
        """
        if reasoning_effort is None or reasoning_effort == '':
            return False
        
        reasoning_effort_lower = reasoning_effort.lower()
        
        # 只有在启用思考时才返回思考总结
        if reasoning_effort_lower in ['none', 'minimal']:
            return False
        else:
            return True
    
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
        Gemini模型的completion调用
        
        Args:
            model: 模型名称
            messages: 消息列表
            reasoning_effort: 推理强度
                - None: 使用默认行为（动态思考）
                - "none"/"minimal": 停用思考（Gemini 2.5）或最小思考（Gemini 3）
                - "low": 低强度思考
                - "medium": 中等强度思考（会转换为low并提示）
                - "high": 高强度思考
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            return_full_response: 是否返回完整的response对象
            video_input_mode: 视频输入模式（Gemini只支持"upload"模式）
            **kwargs: 其他参数
            
        Returns:
            - 当return_full_response=False时：
                - 如果没有思考内容：返回字符串（输出内容）
                - 如果有思考内容：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回完整的API响应对象
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # 验证并标准化video_input_mode（Gemini只支持upload）
        video_input_mode = self._validate_video_input_mode(video_input_mode)
        
        # 转换消息格式：从OpenAI风格转为Gemini风格
        contents = self._convert_messages_to_contents(messages)
        
        # 构建配置
        config_params = {}
        
        # 根据模型类型设置思考参数
        if self._is_gemini_3_model(model):
            # Gemini 3使用thinkingLevel
            thinking_level = self._convert_reasoning_effort_to_thinking_level(reasoning_effort)
            include_thoughts = self._should_include_thoughts(reasoning_effort)
            
            if thinking_level or include_thoughts:
                thinking_config_params = {}
                if thinking_level:
                    thinking_config_params['thinking_level'] = thinking_level
                if include_thoughts:
                    thinking_config_params['include_thoughts'] = True
                
                config_params['thinking_config'] = types.ThinkingConfig(**thinking_config_params)
        
        elif self._is_gemini_2_5_model(model):
            # Gemini 2.5使用thinkingBudget
            thinking_budget = self._convert_reasoning_effort_to_thinking_budget(reasoning_effort, model)
            include_thoughts = self._should_include_thoughts(reasoning_effort)
            
            if thinking_budget is not None or include_thoughts:
                thinking_config_params = {}
                if thinking_budget is not None:
                    thinking_config_params['thinking_budget'] = thinking_budget
                if include_thoughts:
                    thinking_config_params['include_thoughts'] = True
                
                config_params['thinking_config'] = types.ThinkingConfig(**thinking_config_params)
        
        # 添加其他可选参数
        if temperature is not None:
            config_params['temperature'] = temperature
        if max_tokens is not None:
            config_params['max_output_tokens'] = max_tokens
        if top_p is not None:
            config_params['top_p'] = top_p
        
        # 创建配置对象
        if config_params:
            config = types.GenerateContentConfig(**config_params)
        else:
            config = None
        
        # 调用API
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        # 解析响应
        thought_summary = ""
        answer_text = ""
        
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thought_summary += part.text
            else:
                answer_text += part.text
        
        # 根据return_full_response决定返回格式
        if return_full_response:
            # 创建一个包装类来增强response对象
            class EnhancedResponse:
                def __init__(self, original_response, thinking_content):
                    self._original = original_response
                    self.thinking_content = thinking_content
                
                def __getattr__(self, name):
                    # 代理访问原始response的属性
                    return getattr(self._original, name)
            
            return EnhancedResponse(response, thought_summary if thought_summary else None)
        else:
            # 返回解析后的内容
            if thought_summary:
                return {
                    "content": answer_text,
                    "thinking_content": thought_summary
                }
            else:
                return answer_text
    
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
        Gemini模型的异步completion调用
        
        Args:
            model: 模型名称
            messages: 消息列表
            reasoning_effort: 推理强度
                - None: 使用默认行为（动态思考）
                - "none"/"minimal": 停用思考（Gemini 2.5）或最小思考（Gemini 3）
                - "low": 低强度思考
                - "medium": 中等强度思考（会转换为low并提示）
                - "high": 高强度思考
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: nucleus sampling参数
            return_full_response: 是否返回完整的response对象
            video_input_mode: 视频输入模式（Gemini只支持"upload"模式）
            **kwargs: 其他参数
            
        Returns:
            - 当return_full_response=False时：
                - 如果没有思考内容：返回字符串（输出内容）
                - 如果有思考内容：返回字典 {"content": "...", "thinking_content": "..."}
            - 当return_full_response=True时：返回完整的API响应对象
        """
        if not self.supports_model(model):
            raise ValueError(f"不支持的模型: {model}. 支持的模型: {', '.join(self.SUPPORTED_MODELS)}")
        
        # 验证并标准化video_input_mode（Gemini只支持upload）
        video_input_mode = self._validate_video_input_mode(video_input_mode)
        
        # 转换消息格式：从OpenAI风格转为Gemini风格
        contents = self._convert_messages_to_contents(messages)
        
        # 构建配置
        config_params = {}
        
        # 根据模型类型设置思考参数
        if self._is_gemini_3_model(model):
            # Gemini 3使用thinkingLevel
            thinking_level = self._convert_reasoning_effort_to_thinking_level(reasoning_effort)
            include_thoughts = self._should_include_thoughts(reasoning_effort)
            
            if thinking_level or include_thoughts:
                thinking_config_params = {}
                if thinking_level:
                    thinking_config_params['thinking_level'] = thinking_level
                if include_thoughts:
                    thinking_config_params['include_thoughts'] = True
                
                config_params['thinking_config'] = types.ThinkingConfig(**thinking_config_params)
        
        elif self._is_gemini_2_5_model(model):
            # Gemini 2.5使用thinkingBudget
            thinking_budget = self._convert_reasoning_effort_to_thinking_budget(reasoning_effort, model)
            include_thoughts = self._should_include_thoughts(reasoning_effort)
            
            if thinking_budget is not None or include_thoughts:
                thinking_config_params = {}
                if thinking_budget is not None:
                    thinking_config_params['thinking_budget'] = thinking_budget
                if include_thoughts:
                    thinking_config_params['include_thoughts'] = True
                
                config_params['thinking_config'] = types.ThinkingConfig(**thinking_config_params)
        
        # 添加其他可选参数
        if temperature is not None:
            config_params['temperature'] = temperature
        if max_tokens is not None:
            config_params['max_output_tokens'] = max_tokens
        if top_p is not None:
            config_params['top_p'] = top_p
        
        # 创建配置对象
        if config_params:
            config = types.GenerateContentConfig(**config_params)
        else:
            config = None
        
        # 异步调用API（使用 client.aio.models.generate_content）
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        # 解析响应
        thought_summary = ""
        answer_text = ""
        
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thought_summary += part.text
            else:
                answer_text += part.text
        
        # 根据return_full_response决定返回格式
        if return_full_response:
            # 创建一个包装类来增强response对象
            class EnhancedResponse:
                def __init__(self, original_response, thinking_content):
                    self._original = original_response
                    self.thinking_content = thinking_content
                
                def __getattr__(self, name):
                    # 代理访问原始response的属性
                    return getattr(self._original, name)
            
            return EnhancedResponse(response, thought_summary if thought_summary else None)
        else:
            # 返回解析后的内容
            if thought_summary:
                return {
                    "content": answer_text,
                    "thinking_content": thought_summary
                }
            else:
                return answer_text

