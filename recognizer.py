"""
Face recognition module using InsightFace.
Handles face embedding generation and recognition.
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional
import insightface
from sklearn.metrics.pairwise import cosine_similarity
import uuid

class FaceRecognizer:
    def __init__(self, model_name: str = 'buffalo_l', similarity_threshold: float = 0.6):
        """
        Initialize InsightFace recognition model.
        
        Args:
            model_name: InsightFace model name (buffalo_l, buffalo_m, buffalo_s)
            similarity_threshold: Threshold for face matching
        """
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.load_model(model_name)
        
    def load_model(self, model_name: str):
        """Load InsightFace model."""
        try:
            self.model = insightface.app.FaceAnalysis(name=model_name)
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            logging.info(f"Loaded InsightFace model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading InsightFace model: {e}")
            # Fallback to default model
            try:
                self.model = insightface.app.FaceAnalysis()
                self.model.prepare(ctx_id=0, det_size=(640, 640))
                logging.info("Loaded default InsightFace model")
            except Exception as e2:
                logging.error(f"Error loading default model: {e2}")
                raise
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding from cropped face image.
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            Face embedding vector or None if failed
        """
        if self.model is None:
            logging.error("Recognition model not loaded")
            return None
        
        try:
            # Ensure image is in correct format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Convert RGB to BGR if needed (InsightFace expects BGR)
                if face_image.dtype != np.uint8:
                    face_image = (face_image * 255).astype(np.uint8)
            else:
                logging.error("Invalid face image format")
                return None
            
            # Get face analysis
            faces = self.model.get(face_image)
            
            if len(faces) == 0:
                logging.warning("No face detected in cropped image")
                return None
            
            # Use the first (and hopefully only) face
            face = faces[0]
            embedding = face.embedding
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Error generating face embedding: {e}")
            return None
    
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
            # Reshape for sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
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
    
    def batch_recognize_faces(self, face_images: List[np.ndarray], 
                            known_embeddings: List[Tuple[str, np.ndarray]]) -> List[Tuple[Optional[str], float, Optional[np.ndarray]]]:
        """
        Recognize multiple faces in batch.
        
        Args:
            face_images: List of cropped face images
            known_embeddings: List of (face_id, embedding) tuples
            
        Returns:
            List of recognition results for each face
        """
        results = []
        
        for face_image in face_images:
            result = self.recognize_face(face_image, known_embeddings)
            results.append(result)
        
        return results
    
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
            if self.model is None:
                return 0.0
            
            faces = self.model.get(face_image)
            
            if len(faces) == 0:
                return 0.0
            
            face = faces[0]
            
            # Combine multiple quality factors
            quality_factors = []
            
            # Face detection confidence
            if hasattr(face, 'det_score'):
                quality_factors.append(face.det_score)
            
            # Face size (larger faces are generally better)
            bbox = face.bbox
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            img_area = face_image.shape[0] * face_image.shape[1]
            size_ratio = min(1.0, face_area / (img_area * 0.1))  # Normalize
            quality_factors.append(size_ratio)
            
            # Image sharpness (Laplacian variance)
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)  # Normalize
            quality_factors.append(sharpness_score)
            
            # Average the quality factors
            if quality_factors:
                return np.mean(quality_factors)
            else:
                return 0.5  # Default moderate quality
                
        except Exception as e:
            logging.error(f"Error calculating face quality: {e}")
            return 0.0
    
    def is_good_quality_face(self, face_image: np.ndarray, min_quality: float = 0.3) -> bool:
        """
        Check if face image has sufficient quality for recognition.
        
        Args:
            face_image: Cropped face image
            min_quality: Minimum quality threshold
            
        Returns:
            True if face quality is acceptable
        """
        quality_score = self.get_face_quality_score(face_image)
        return quality_score >= min_quality
