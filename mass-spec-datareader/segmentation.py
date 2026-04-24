import cv2
import numpy as np
import os

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

class BIOSegmenter:
    """
    Intelligent tissue segmenter for chemical biology research.
    Provides a 'Graceful Upgrade' from classic math to MedSAM.
    """
    
    def __init__(self, model_type="vit_b", checkpoint_path="models/medsam.pth"):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.predictor = None
        
        # Auto-initialize SAM if weights exist
        if HAS_SAM and os.path.exists(checkpoint_path):
            try:
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam.to(device=device)
                self.predictor = SamPredictor(sam)
                print(f"--- MedSAM Engine Active ({device}) ---")
            except Exception as e:
                print(f"Warning: Failed to load MedSAM: {e}. Falling back to classic math.")

    def otsu_mask(self, image):
        """Classic morphological masking (Zero-install, high performance)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 1. Blur to remove noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Otsu Automated Thresholding
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Morphological Closing to fill holes (Tissue is often porous)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. Remove small floating artifacts
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255
            
        return mask

    def sam_mask(self, image, box=None):
        """Model-powered segmentation using Segment Anything."""
        if self.predictor is None:
            return self.otsu_mask(image)
            
        self.predictor.set_image(image)
        
        # If no box provided, we assume the user wants the whole tissue structure
        # In a production GUI, the user would provide a bounding box
        if box is None:
            h, w = image.shape[:2]
            box = np.array([0, 0, w, h])
            
        masks, scores, logits = self.predictor.predict(
            box=box[None, :],
            multimask_output=False,
        )
        return (masks[0] * 255).astype(np.uint8)

    def auto_mask(self, image):
        """The 'Easy Button' for researchers."""
        if self.predictor:
            return self.sam_mask(image)
        return self.otsu_mask(image)

