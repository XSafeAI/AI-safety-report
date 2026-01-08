"""
LLM Provider base class definition
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLMProvider(ABC):
    """LLM Provider base class"""
    
    # Subclasses must define supported model list
    SUPPORTED_MODELS: List[str] = []
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Provider
        
        Args:
            api_key: API key, reads from environment variables if not provided
            **kwargs: Other configuration parameters
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
        Unified completion interface
        
        Args:
            model: Model name
            messages: Message list
            return_full_response: Whether to return full response object, default False returns parsed text
            **kwargs: Other parameters
            
        Returns:
            - When return_full_response=False:
                - For normal models: returns string (output content)
                - For models with thinking content: returns dictionary {"content": "...", "thinking_content": "..."}
            - When return_full_response=True: returns full API response object
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
        Unified async completion interface
        
        Args:
            model: Model name
            messages: Message list
            return_full_response: Whether to return full response object, default False returns parsed text
            **kwargs: Other parameters
            
        Returns:
            - When return_full_response=False:
                - For normal models: returns string (output content)
                - For models with thinking content: returns dictionary {"content": "...", "thinking_content": "..."}
            - When return_full_response=True: returns full API response object
        """
        pass
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """
        Get list of supported models (class method, can be called before instantiation)
        
        Returns:
            List of supported models
        """
        return cls.SUPPORTED_MODELS
    
    def supports_model(self, model: str) -> bool:
        """
        Determine if the model is supported
        
        Args:
            model: Model name
            
        Returns:
            Whether supported
        """
        # Supports exact match or prefix match
        return any(model.startswith(supported) or model == supported 
                   for supported in self.SUPPORTED_MODELS)

