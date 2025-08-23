"""Helper utilities for etnltk integration."""

import logging
from typing import List

logger = logging.getLogger(__name__)

class EtnltkHelper:
    """Helper class for etnltk operations."""
    
    def __init__(self):
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if etnltk is available."""
        try:
            import etnltk
            return True
        except ImportError:
            logger.warning("etnltk not available. Install with: pip install etnltk")
            return False
    
    def tokenize(self, text: str, lang: str = 'am') -> List[str]:
        """Tokenize text using etnltk."""
        if not self.available:
            return text.split()
            
        try:
            from etnltk.tokenize import word_tokenize
            return word_tokenize(text, lang=lang)
        except Exception as e:
            logger.warning(f"etnltk tokenization failed: {e}")
            return text.split()
    
    def normalize(self, text: str) -> str:
        """Normalize Amharic text using etnltk."""
        if not self.available:
            return text
            
        try:
            from etnltk.lang.am import normalize
            return normalize(text)
        except Exception as e:
            logger.warning(f"etnltk normalization failed: {e}")
            return text

# Global instance
etnltk_helper = EtnltkHelper()