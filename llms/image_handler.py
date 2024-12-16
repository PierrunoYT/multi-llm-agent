import base64
from pathlib import Path
from typing import Optional
import mimetypes
import imghdr
from PIL import Image
import io

class ImageHandler:
    """Handles image processing and validation for LLM requests."""
    
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
    SUPPORTED_FORMATS = {'jpeg', 'png', 'gif', 'webp'}
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """
        Validate image file format and size.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if image is valid, False otherwise
            
        Raises:
            ValueError: If image is invalid with specific reason
        """
        path = Path(image_path)
        
        # Check if file exists
        if not path.exists():
            raise ValueError(f"Image file not found: {image_path}")
            
        # Check file size
        if path.stat().st_size > ImageHandler.MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image size exceeds maximum allowed size of "
                f"{ImageHandler.MAX_IMAGE_SIZE / 1024 / 1024}MB"
            )
            
        # Verify image format
        img_format = imghdr.what(image_path)
        if not img_format or img_format not in ImageHandler.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format. Supported formats: "
                f"{', '.join(ImageHandler.SUPPORTED_FORMATS)}"
            )
            
        return True
    
    @staticmethod
    def encode_image(image_path: str, max_dimension: Optional[int] = 2048) -> str:
        """
        Encode image to base64 with optional resizing.
        
        Args:
            image_path: Path to the image file
            max_dimension: Maximum dimension (width/height) for resizing
            
        Returns:
            str: Base64 encoded image with data URI scheme
            
        Raises:
            ValueError: If image processing fails
        """
        try:
            # Validate image
            ImageHandler.validate_image(image_path)
            
            # Open and optionally resize image
            with Image.open(image_path) as img:
                if max_dimension:
                    # Calculate new dimensions maintaining aspect ratio
                    ratio = min(max_dimension / max(img.size))
                    if ratio < 1:
                        new_size = tuple(int(dim * ratio) for dim in img.size)
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                image_data = buffer.getvalue()
            
            # Encode to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            mime_type = mimetypes.guess_type(image_path)[0] or 'image/jpeg'
            
            return f"data:{mime_type};base64,{base64_image}"
            
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}") from e
    
    @staticmethod
    def get_image_metadata(image_path: str) -> dict:
        """
        Get image metadata.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Image metadata including dimensions, format, and size
        """
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format.lower(),
                'mode': img.mode,
                'size_bytes': Path(image_path).stat().st_size
            }
