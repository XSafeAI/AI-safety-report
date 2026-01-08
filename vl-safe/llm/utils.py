"""
LLM Provider 通用工具函数
"""

import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


def is_url(path: str) -> bool:
    """
    判断是否为URL
    
    Args:
        path: 路径字符串
        
    Returns:
        是否为URL
    """
    if not isinstance(path, str):
        return False
    return path.startswith(('http://', 'https://'))


def is_base64_data_url(data: str) -> bool:
    """
    判断是否为Base64 Data URL格式（data:image/jpeg;base64,...）
    
    Args:
        data: 数据字符串
        
    Returns:
        是否为Base64 Data URL
    """
    if not isinstance(data, str):
        return False
    return data.startswith('data:')


def is_local_file(path: str) -> bool:
    """
    判断是否为本地文件路径
    
    Args:
        path: 路径字符串
        
    Returns:
        是否为本地文件且存在
    """
    if not isinstance(path, str):
        return False
    if is_url(path) or is_base64_data_url(path):
        return False
    return Path(path).exists()


def get_mime_type(url_or_path: str, content_type: Optional[str] = None) -> str:
    """
    获取文件的MIME类型
    
    Args:
        url_or_path: URL或文件路径
        content_type: 内容类型提示（如 "image_url", "video_url"）
        
    Returns:
        MIME类型字符串
    """
    if content_type:
        if content_type == "image_url":
            return "image/jpeg"  # 默认图片类型
        elif content_type == "video_url":
            return "video/mp4"   # 默认视频类型
    
    # 从文件扩展名推断
    ext = Path(url_or_path).suffix.lower()
    mime_types = {
        # 图片格式
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        # 视频格式
        '.mp4': 'video/mp4',
        '.avi': 'video/avi',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm',
        '.mkv': 'video/x-matroska',
        # 音频格式
        '.mp3': 'audio/mp3',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
    }
    return mime_types.get(ext, 'image/jpeg')


def get_image_size(image_path: str) -> int:
    """
    获取图片文件大小（字节）
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        文件大小（字节）
    """
    return Path(image_path).stat().st_size


def compress_image(
    input_path: str,
    output_path: str,
    target_size_mb: float,
    quality: int = 85
) -> str:
    """
    压缩图片到目标大小以下
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        target_size_mb: 目标大小（MB）
        quality: JPEG质量（1-100）
        
    Returns:
        压缩后的图片路径
    """
    import tempfile
    from PIL import Image
    import os
    
    # 打开图片
    img = Image.open(input_path)
    
    # 转换为RGB模式（如果是RGBA或其他模式）
    if img.mode in ('RGBA', 'LA', 'P'):
        # 创建白色背景
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    target_size_bytes = target_size_mb * 1024 * 1024
    
    # 尝试不同的质量参数
    for q in range(quality, 10, -10):
        img.save(output_path, 'JPEG', quality=q, optimize=True)
        
        output_size = os.path.getsize(output_path)
        if output_size <= target_size_bytes:
            return output_path
    
    # 如果质量降到最低还是太大，尝试缩小尺寸
    scale_factor = 0.8
    while True:
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        
        if new_width < 100 or new_height < 100:
            # 已经太小了，无法继续缩小
            break
        
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_img.save(output_path, 'JPEG', quality=70, optimize=True)
        
        output_size = os.path.getsize(output_path)
        if output_size <= target_size_bytes:
            return output_path
        
        scale_factor *= 0.8
    
    return output_path


def compress_image_if_needed(
    image_path: str,
    max_size_mb: float,
    warning_callback = None
) -> str:
    """
    如果图片超过指定大小，则压缩图片
    
    Args:
        image_path: 图片文件路径
        max_size_mb: 最大允许大小（MB）
        warning_callback: 警告回调函数
        
    Returns:
        处理后的图片路径（可能是原始路径或压缩后的临时文件路径）
    """
    import tempfile
    import warnings
    import os
    
    image_size = get_image_size(image_path)
    image_size_mb = image_size / (1024 * 1024)
    
    if image_size_mb <= max_size_mb:
        return image_path
    
    # 需要压缩
    warning_msg = f"图片大小 {image_size_mb:.2f}MB 超过限制 {max_size_mb}MB，正在压缩..."
    if warning_callback:
        warning_callback(warning_msg)
    else:
        warnings.warn(warning_msg, UserWarning)
    
    # 创建临时文件
    temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
    os.close(temp_fd)
    
    try:
        compress_image(image_path, temp_path, max_size_mb * 0.9)  # 压缩到90%以确保安全
        
        compressed_size = get_image_size(temp_path)
        compressed_size_mb = compressed_size / (1024 * 1024)
        
        success_msg = f"图片已压缩: {image_size_mb:.2f}MB -> {compressed_size_mb:.2f}MB"
        if warning_callback:
            warning_callback(success_msg)
        else:
            warnings.warn(success_msg, UserWarning)
        
        return temp_path
    except Exception as e:
        # 压缩失败，删除临时文件
        if Path(temp_path).exists():
            os.unlink(temp_path)
        raise e


def local_image_to_base64(file_path: str, include_mime: bool = True) -> str:
    """
    将本地图片文件转换为Base64格式
    
    Args:
        file_path: 本地文件路径
        include_mime: 是否包含MIME类型前缀（data:image/jpeg;base64,...）
        
    Returns:
        Base64编码的字符串
    """
    with open(file_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    if include_mime:
        mime_type = get_mime_type(file_path, 'image_url')
        return f"data:{mime_type};base64,{image_data}"
    else:
        return image_data


def local_video_to_base64(file_path: str, include_mime: bool = True) -> str:
    """
    将本地视频文件转换为Base64格式
    
    Args:
        file_path: 本地文件路径
        include_mime: 是否包含MIME类型前缀（data:video/mp4;base64,...）
        
    Returns:
        Base64编码的字符串
    """
    with open(file_path, 'rb') as f:
        video_data = base64.b64encode(f.read()).decode('utf-8')
    
    if include_mime:
        mime_type = get_mime_type(file_path, 'video_url')
        return f"data:{mime_type};base64,{video_data}"
    else:
        return video_data


def extract_frames_from_video(
    video_path: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None
) -> List[np.ndarray]:
    """
    从视频中抽取帧
    
    Args:
        video_path: 视频文件路径
        fps: 每秒抽取的帧数（如果为None，使用max_frames）
        max_frames: 最大抽取帧数（如果为None，使用fps）
        
    Returns:
        抽取的帧列表（numpy数组）
        
    Note:
        - 如果fps和max_frames都提供，选择产生帧数最少的方式
        - 如果都不提供，默认fps=2
    """
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    # 决定抽帧策略
    if fps is None and max_frames is None:
        # 默认使用fps=2
        fps = 2.0
    
    # 计算实际要抽取的帧
    frames_to_extract = []
    
    if fps is not None and max_frames is not None:
        # 两个都提供，选择产生帧数最少的
        frames_by_fps = int(duration * fps)
        if frames_by_fps <= max_frames:
            # 使用fps方式
            frame_interval = video_fps / fps
            frames_to_extract = [int(i * frame_interval) for i in range(frames_by_fps)]
        else:
            # 使用max_frames方式
            frame_interval = total_frames / max_frames
            frames_to_extract = [int(i * frame_interval) for i in range(max_frames)]
    elif fps is not None:
        # 只提供fps
        frames_by_fps = int(duration * fps)
        frame_interval = video_fps / fps
        frames_to_extract = [int(i * frame_interval) for i in range(frames_by_fps)]
    else:
        # 只提供max_frames
        frame_interval = total_frames / max_frames
        frames_to_extract = [int(i * frame_interval) for i in range(max_frames)]
    
    # 确保不超过总帧数
    frames_to_extract = [f for f in frames_to_extract if f < total_frames]
    
    # 抽取帧
    extracted_frames = []
    for frame_idx in frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            extracted_frames.append(frame)
    
    cap.release()
    return extracted_frames


def frame_to_base64(frame: np.ndarray, format: str = 'JPEG') -> str:
    """
    将视频帧（numpy数组）转换为Base64格式
    
    Args:
        frame: 视频帧（numpy数组）
        format: 图片格式（JPEG或PNG）
        
    Returns:
        Base64编码的图片字符串（带data URL前缀）
    """
    # 将BGR转为RGB（OpenCV使用BGR，但一般图片是RGB）
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    # 编码为JPEG或PNG
    if format.upper() == 'JPEG':
        success, buffer = cv2.imencode('.jpg', frame_rgb)
        mime_type = 'image/jpeg'
    else:
        success, buffer = cv2.imencode('.png', frame_rgb)
        mime_type = 'image/png'
    
    if not success:
        raise ValueError("无法编码帧为图片格式")
    
    # 转换为Base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:{mime_type};base64,{img_base64}"


def extract_video_frames_as_base64(
    video_path: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    format: str = 'JPEG'
) -> List[str]:
    """
    从视频中抽取帧并转换为Base64格式
    
    Args:
        video_path: 视频文件路径
        fps: 每秒抽取的帧数
        max_frames: 最大抽取帧数
        format: 图片格式（JPEG或PNG）
        
    Returns:
        Base64编码的图片字符串列表
    """
    frames = extract_frames_from_video(video_path, fps, max_frames)
    return [frame_to_base64(frame, format) for frame in frames]


def get_video_size(video_path: str) -> int:
    """
    获取视频文件大小（字节）
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        文件大小（字节）
    """
    return Path(video_path).stat().st_size


def compress_video(
    input_path: str,
    output_path: str,
    target_size_mb: float,
    max_attempts: int = 3
) -> str:
    """
    压缩视频到目标大小以下
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        target_size_mb: 目标大小（MB）
        max_attempts: 最大尝试次数
        
    Returns:
        压缩后的视频路径
    """
    import subprocess
    import os
    
    # 获取原始视频信息
    cap = cv2.VideoCapture(input_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    cap.release()
    
    # 计算目标码率（比特率）
    target_size_bits = target_size_mb * 8 * 1024 * 1024
    target_bitrate = int(target_size_bits / duration) if duration > 0 else 500000
    
    # 尝试不同的压缩参数
    for attempt in range(max_attempts):
        # 每次尝试降低码率
        bitrate = int(target_bitrate * (0.8 ** attempt))
        
        # 使用ffmpeg压缩视频
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-i', input_path,
            '-c:v', 'libx264',  # 使用H.264编码
            '-b:v', f'{bitrate}',  # 视频码率
            '-preset', 'fast',  # 编码速度
            '-c:a', 'aac',  # 音频编码
            '-b:a', '128k',  # 音频码率
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 检查输出文件大小
            output_size = get_video_size(output_path)
            output_size_mb = output_size / (1024 * 1024)
            
            if output_size_mb <= target_size_mb:
                return output_path
        except subprocess.CalledProcessError:
            continue
    
    # 如果还是太大，尝试降低分辨率
    scale_factor = 0.7
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    cmd = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-b:v', f'{int(target_bitrate * 0.5)}',
        '-vf', f'scale={new_width}:{new_height}',
        '-preset', 'fast',
        '-c:a', 'aac',
        '-b:a', '96k',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"视频压缩失败: {e}")


def compress_video_if_needed(
    video_path: str,
    max_size_mb: float,
    warning_callback = None
) -> str:
    """
    如果视频超过指定大小，则压缩视频
    
    Args:
        video_path: 视频文件路径
        max_size_mb: 最大允许大小（MB）
        warning_callback: 警告回调函数
        
    Returns:
        处理后的视频路径（可能是原始路径或压缩后的临时文件路径）
    """
    import tempfile
    import warnings
    
    video_size = get_video_size(video_path)
    video_size_mb = video_size / (1024 * 1024)
    
    if video_size_mb <= max_size_mb:
        return video_path
    
    # 需要压缩
    warning_msg = f"视频大小 {video_size_mb:.2f}MB 超过限制 {max_size_mb}MB，正在压缩..."
    if warning_callback:
        warning_callback(warning_msg)
    else:
        warnings.warn(warning_msg, UserWarning)
    
    # 创建临时文件
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
    import os
    os.close(temp_fd)
    
    try:
        compress_video(video_path, temp_path, max_size_mb * 0.9)  # 压缩到90%以确保安全
        
        compressed_size = get_video_size(temp_path)
        compressed_size_mb = compressed_size / (1024 * 1024)
        
        success_msg = f"视频已压缩: {video_size_mb:.2f}MB -> {compressed_size_mb:.2f}MB"
        if warning_callback:
            warning_callback(success_msg)
        else:
            warnings.warn(success_msg, UserWarning)
        
        return temp_path
    except Exception as e:
        # 压缩失败，删除临时文件
        if Path(temp_path).exists():
            os.unlink(temp_path)
        raise e

