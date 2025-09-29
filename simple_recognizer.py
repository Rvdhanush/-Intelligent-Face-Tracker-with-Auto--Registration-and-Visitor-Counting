"""
Simplified face recognition module without InsightFace dependency.
Uses basic face features for demonstration purposes.
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional
import uuid
import hashlib

class SimpleFaceRecognizer:
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize simple face recognizer.
        
        Args:
            similarity_threshold: Threshold for face matching
        """
        self.similarity_threshold = similarity_threshold
        logging.info("Simple face recognizer initialized")
        
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate simple face embedding from cropped face image.
        Uses basic image features like histogram and texture.
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            Simple feature vector or None if failed
        """
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_image, (64, 64))
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram features
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-7)  # Normalize
            
            # Calculate LBP (Local Binary Pattern) features
            lbp = self._calculate_lbp(gray)
            lbp_hist = cv2.calcHist([lbp], [0], None, [32], [0, 256])
            lbp_hist = lbp_hist.flatten()
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-7)  # Normalize
            
            # Calculate edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Combine features
            embedding = np.concatenate([hist, lbp_hist, [edge_density]])
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error generating simple embedding: {e}")
            return None
    
    def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        rows, cols = gray_image.shape
        lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = gray_image[i, j]
                code = 0
                
                # 8-neighborhood
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i-1, j-1] = code
        
        return lbp
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logging.error(f"Error comparing embeddings: {e}")
            return 0.0
    
    def find_best_match(self, query_embedding: np.ndarray, 
                       known_embeddings: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float]:
        """
        Find the best matching face from known embeddings.
        
        Args:
            query_embedding: Embedding to match
            known_embeddings: List of (face_id, embedding) tuples
            
        Returns:
            Tuple of (best_match_id, similarity_score) or (None, 0.0) if no match
        """
        if not known_embeddings:
            return None, 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        try:
            for face_id, known_embedding in known_embeddings:
                similarity = self.compare_embeddings(query_embedding, known_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = face_id
            
            # Check if best match meets threshold
            if best_similarity >= self.similarity_threshold:
                return best_match_id, best_similarity
            else:
                return None, best_similarity
                
        except Exception as e:
            logging.error(f"Error finding best match: {e}")
            return None, 0.0
    
    def recognize_face(self, face_image: np.ndarray, 
                      known_embeddings: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """
        Recognize a face by comparing with known embeddings.
        
        Args:
            face_image: Cropped face image
            known_embeddings: List of (face_id, embedding) tuples
            
        Returns:
            Tuple of (face_id, confidence, embedding) or (None, 0.0, embedding) for new face
        """
        # Generate embedding for the input face
        embedding = self.generate_embedding(face_image)
        
        if embedding is None:
            return None, 0.0, None
        
        # Find best match
        best_match_id, similarity = self.find_best_match(embedding, known_embeddings)
        
        return best_match_id, similarity, embedding
    
    def generate_face_id(self) -> str:
        """Generate a unique face ID."""
        return f"face_{uuid.uuid4().hex[:8]}"
    
    def is_good_quality_face(self, face_image: np.ndarray, min_quality: float = 0.3) -> bool:
        """
        Check if face image has sufficient quality for recognition.
        
        Args:
            face_image: Cropped face image
            min_quality: Minimum quality threshold
            
        Returns:
            True if face quality is acceptable
        """
        try:
            # Simple quality checks
            if face_image is None or face_image.size == 0:
                return False
            
            # Check size
            if face_image.shape[0] < 32 or face_image.shape[1] < 32:
                return False
            
            # Check brightness
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < 30 or mean_brightness > 220:
                return False
            
            # Check contrast (standard deviation)
            contrast = np.std(gray)
            if contrast < 20:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking face quality: {e}")
            return False
    
    def update_similarity_threshold(self, new_threshold: float):
        """Update the similarity threshold for face matching."""
        if 0.0 <= new_threshold <= 1.0:
            self.similarity_threshold = new_threshold
            logging.info(f"Updated similarity threshold to {new_threshold}")
        else:
            logging.warning(f"Invalid threshold value: {new_threshold}")
    
    def get_face_quality_score(self, face_image: np.ndarray) -> float:
        """
        Assess the quality of a face image for recognition.
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Quality score (0-1, higher is better)
        """
        try:
            if face_image is None or face_image.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Brightness score
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            
            # Contrast score
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 50.0)
            
            # Sharpness score (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)
            
            # Size score
            size_score = min(1.0, min(face_image.shape[:2]) / 64.0)
            
            # Combined score
            quality_score = (brightness_score + contrast_score + sharpness_score + size_score) / 4.0
            
            return quality_score
            
        except Exception as e:
            logging.error(f"Error calculating face quality: {e}")
            return 0.0
