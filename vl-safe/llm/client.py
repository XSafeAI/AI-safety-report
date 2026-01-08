"""
Unified LLM client
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
    """Unified LLM calling client (using lazy loading pattern)"""
    
    def __init__(self, **provider_configs):
        """
        Initialize LLM client
        
        Args:
            **provider_configs: Configuration for each provider
                For example: openai_api_key="xxx", qwen_api_key="yyy"
        
        Note: Using lazy loading pattern, provider will only be initialized on first use
        """
        self.provider_configs = provider_configs
        
        # Use dictionary to store initialized provider instances (lazy loading)
        self._initialized_providers: Dict[str, BaseLLMProvider] = {}
        
        # Provider type mapping table (defines supported provider types)
        self._provider_classes = {
            'openai': OpenAIProvider,
            'ark': ArkProvider,
            'dashscope': DashScopeProvider,
            'gemini': GeminiProvider,
            'deepseek': DeepSeekProvider,
            'siliconflow': SiliconFlowProvider,
            'xai': XAIProvider,
        }
        
        # Automatically build model to provider mapping table
        self._model_to_provider_map = self._build_model_mapping()
    
    def _build_model_mapping(self) -> Dict[str, str]:
        """
        Automatically build mapping table from models to provider types
        Get supported model list from each provider class and build mapping relationship
        
        Returns:
            Mapping dictionary from model prefix to provider type
        """
        mapping = {}
        for provider_type, provider_class in self._provider_classes.items():
            # Get supported model list for this provider
            supported_models = provider_class.get_supported_models()
            for model in supported_models:
                # Use model name as key, provider type as value
                mapping[model] = provider_type
        return mapping
    
    def _get_provider_for_model(self, model: str) -> Optional[str]:
        """
        Determine which provider should be used based on model name
        Use automatically built mapping table for lookup
        
        Args:
            model: Model name
            
        Returns:
            Provider type name, returns None if not found
        """
        if model in self._model_to_provider_map:
            return self._model_to_provider_map[model]
        
        return None
    
    def _get_or_create_provider(self, provider_type: str) -> BaseLLMProvider:
        """
        Get or create provider instance (lazy loading)
        
        Args:
            provider_type: Provider type name
            
        Returns:
            Provider instance
        """
        # If already initialized, return directly
        if provider_type in self._initialized_providers:
            return self._initialized_providers[provider_type]
        
        # Get provider class
        provider_class = self._provider_classes.get(provider_type)
        if provider_class is None:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        # Extract configuration parameters for this provider
        api_key = self.provider_configs.get(f'{provider_type}_api_key')
        provider_specific_config = {
            k.replace(f'{provider_type}_', ''): v 
            for k, v in self.provider_configs.items() 
            if k.startswith(f'{provider_type}_') and k != f'{provider_type}_api_key'
        }
        
        # Create provider instance
        provider = provider_class(api_key=api_key, **provider_specific_config)
        
        # Cache instance
        self._initialized_providers[provider_type] = provider
        
        return provider
    
    def _get_provider(self, model: str) -> Optional[BaseLLMProvider]:
        """
        Automatically select corresponding provider based on model name (lazy loading)
        
        Args:
            model: Model name
            
        Returns:
            Corresponding provider, returns None if not found
        """
        # Determine which provider should be used
        provider_type = self._get_provider_for_model(model)
        if provider_type is None:
            return None
        
        # Lazy load get or create provider
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
        Unified completion interface
        
        Args:
            model: Model name
            messages: Message list
            return_full_response: Whether to return full response object, default False returns parsed content
            video_input_mode: Video input mode, optional values:
                - "base64": Video directly converted to base64 (for ark, dashscope)
                - "frames": Video extracted to multiple images (for openai, ark, dashscope)
                - "upload": Upload file to server (for gemini)
                - None: Use provider's default behavior
            **kwargs: Other parameters (such as reasoning_effort, temperature, etc.)
            
        Returns:
            - When return_full_response=False:
                - For normal models: returns string (output content)
                - For models with thinking content: returns dictionary {"content": "...", "thinking_content": "..."}
            - When return_full_response=True: returns full API response object
            
        Raises:
            ValueError: Raised when model is not supported
        """
        provider = self._get_provider(model)
        
        if provider is None:
            raise ValueError(
                f"Provider not found for model '{model}'. "
                f"Please check if the model name is correct."
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
        Unified async completion interface
        
        Args:
            model: Model name
            messages: Message list
            return_full_response: Whether to return full response object, default False returns parsed content
            video_input_mode: Video input mode, optional values:
                - "base64": Video directly converted to base64 (for ark, dashscope)
                - "frames": Video extracted to multiple images (for openai, ark, dashscope)
                - "upload": Upload file to server (for gemini)
                - None: Use provider's default behavior
            **kwargs: Other parameters (such as reasoning_effort, temperature, etc.)
            
        Returns:
            - When return_full_response=False:
                - For normal models: returns string (output content)
                - For models with thinking content: returns dictionary {"content": "...", "thinking_content": "..."}
            - When return_full_response=True: returns full API response object
            
        Raises:
            ValueError: Raised when model is not supported
        """
        provider = self._get_provider(model)
        
        if provider is None:
            raise ValueError(
                f"Provider not found for model '{model}'. "
                f"Please check if the model name is correct."
            )
        
        return await provider.acompletion(
            model=model, 
            messages=messages, 
            return_full_response=return_full_response,
            video_input_mode=video_input_mode,
            **kwargs
        )


# Create a global default client instance
_default_client = None


def get_default_client() -> LLMClient:
    """Get default LLM client"""
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
    Convenient completion function using default client
    
    Args:
        model: Model name
        messages: Message list
        return_full_response: Whether to return full response object, default False returns parsed content
        video_input_mode: Video input mode ("base64", "frames", "upload" or None)
        **kwargs: Other parameters
        
    Returns:
        - When return_full_response=False:
            - For normal models: returns string (output content)
            - For models with thinking content: returns dictionary {"content": "...", "thinking_content": "..."}
        - When return_full_response=True: returns full API response object
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
    Convenient async completion function using default client
    
    Args:
        model: Model name
        messages: Message list
        return_full_response: Whether to return full response object, default False returns parsed content
        video_input_mode: Video input mode ("base64", "frames", "upload" or None)
        **kwargs: Other parameters
        
    Returns:
        - When return_full_response=False:
            - For normal models: returns string (output content)
            - For models with thinking content: returns dictionary {"content": "...", "thinking_content": "..."}
        - When return_full_response=True: returns full API response object
    """
    client = get_default_client()
    return await client.acompletion(
        model=model, 
        messages=messages, 
        return_full_response=return_full_response,
        video_input_mode=video_input_mode,
        **kwargs
    )

