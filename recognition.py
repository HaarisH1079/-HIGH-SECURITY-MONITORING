import cv2
import numpy as np
import logging
from deepface import DeepFace

class FaceRecognitionEngine:
    def __init__(self, model_name="VGG-Face", detector_backend="opencv", db_manager=None):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.db_manager = db_manager
        self.known_embeddings = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load known faces if DB manager is provided
        if self.db_manager:
            self.reload_embeddings()

    def reload_embeddings(self):
        """Reloads embeddings from the database."""
        if self.db_manager:
            try:
                # The new DB returns {name: {'embedding': [..], 'rank': int, 'id': int}}
                # We need to adapt it for recognition
                users_data = self.db_manager.get_user_embeddings()
                self.known_embeddings = {}
                for name, data in users_data.items():
                    self.known_embeddings[name] = data['embedding']
                print(f"Loaded {len(self.known_embeddings)} known faces.")
            except AttributeError as e:
                print(f"DB Error during reload: {e}")

    def detect_faces(self, frame):
        """
        Detect faces in the frame using Haar Cascade (faster for real-time).
        Returns list of (x, y, w, h) tuples.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def recognize_face(self, face_img):
        """
        Generate embedding for the face image and find the best match.
        Returns: (name, confidence) or ("Unknown", 0.0)
        """
        try:
            # Generate embedding
            # represent() returns a list of dicts. We take the first one.
            embedding_objs = DeepFace.represent(
                img_path=face_img,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend="skip" 
            )
            
            if not embedding_objs:
                return "Unknown", 0.0
            
            target_embedding = embedding_objs[0]["embedding"]
            
            best_match = "Unknown"
            best_score = float("inf") # Lower is better for Euclidean/Cosine usually, but DeepFace.verify uses specific thresholds.
            # We will use Cosine distance manually for speed/simplicity
            
            for name, known_emb in self.known_embeddings.items():
                score = self.find_cosine_distance(target_embedding, known_emb)
                if score < best_score:
                    best_score = score
                    best_match = name
            
            # Threshold for VGG-Face cosine is commonly around 0.4 - 0.5
            # Loosening to 0.6 to catch more faces (user seeing too many Unknowns)
            if best_score < 0.60: 
                return best_match, (1 - best_score) 
            else:
                return "Unknown", 0.0

        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", 0.0

    def find_cosine_distance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def get_embedding(self, face_img):
        """
        Generate embedding for a face image (file path or numpy array).
        Returns the embedding list or None if failed.
        """
        try:
            embedding_objs = DeepFace.represent(
                img_path=face_img,
                model_name=self.model_name,
                enforce_detection=False
            )
            if embedding_objs:
                return embedding_objs[0]["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
        return None
