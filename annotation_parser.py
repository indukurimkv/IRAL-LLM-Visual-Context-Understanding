"""Parser for extracting prefix codes from Roboflow annotations."""

from typing import Dict, Optional


class AnnotationParser:
    """Parser for extracting prefix codes from Roboflow annotations."""
    
    @staticmethod
    def extract_prefix_code(annotation_data: Dict) -> Optional[str]:
        """
        Extract prefix code from annotation.textAnnotations[0].prefix.
        
        Returns the 2-digit code (format: AH) or None if not found.
        """
        try:
            # Check if annotation exists
            if "annotation" not in annotation_data:
                return None
            
            # Navigate to annotation object
            annotation = annotation_data.get("annotation", {})
            if not annotation:
                return None
            
            # Extract textAnnotations array
            text_annotations = annotation.get("textAnnotations", [])
            if not text_annotations:
                return None
            
            # Get prefix code from first text annotation (format: AH where A=anomaly, H=hazard)
            prefix = text_annotations[0].get("prefix")
            return prefix if prefix else None
            
        except (KeyError, IndexError) as e:
            # Handle missing keys or empty lists gracefully
            print(f"Error parsing annotation: {e}")
            return None
