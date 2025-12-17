
import torch
import open_clip
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional

class BioClipClassifier:
    """
    Species classifier using BioClip (Zero-Shot).
    Model: hf-hub:imageomics/bioclip
    """
    
    def __init__(self, species_list: Optional[List[str]] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.species_list = species_list or []
        self.text_features = None
        
        # Initialize model
        self._load_model()
        
        # Pre-compute text features if list provided
        if self.species_list:
            self.update_species_list(self.species_list)
            
    def _load_model(self):
        try:
            print(f"Loading BioClip on {self.device}...")
            # Create model and transforms
            # using 'ViT-B-16-plus-240' pretrained on 'laion400m_e32' usually, 
            # but using the specific HF hub string as requested
            model_name = 'hf-hub:imageomics/bioclip'
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)
            print("BioClip loaded successfully.")
        except Exception as e:
            print(f"Error loading BioClip: {e}")
            self.model = None

    def update_species_list(self, species_list: List[str]):
        """Update the list of species and recompute text embeddings."""
        if not self.model:
            return
            
        self.species_list = species_list
        # Create text prompts (simple class names or templates?)
        # BioClip works well with "a photo of a {species}."
        prompts = [f"a photo of a {species}" for species in species_list]
        
        try:
            text = self.tokenizer(prompts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text)
                self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            print(f"BioClip: Updated text features for {len(species_list)} species.")
        except Exception as e:
            print(f"Error updating species list: {e}")

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Classify the image against the species list.
        Returns (Top Species Name, Confidence Score)
        """
        if not self.model or not self.text_features is not None:
            return ("Unknown", 0.0)
            
        try:
            # Preprocess image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Encode image
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Cosine similarity
                # image_features: [1, D], text_features: [N, D] -> [1, N]
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                
                # Get top 1
                values, indices = similarity[0].topk(1)
                confidence = values.item()
                index = indices.item()
                
                species = self.species_list[index]
                
                return (species, confidence)
                
        except Exception as e:
            print(f"Error in BioClip prediction: {e}")
            return ("Error", 0.0)

    def predict_list(self, image: Image.Image, threshold: float = 0.0, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return list of (species, confidence) tuples above threshold.
        """
        if not self.model or not self.text_features is not None:
             return []
             
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                
                # Get top K
                values, indices = similarity[0].topk(min(top_k, len(self.species_list)))
                
                results = []
                for v, i in zip(values, indices):
                    score = v.item()
                    if score >= threshold:
                        results.append((self.species_list[i.item()], score))
                
                return results
                
        except Exception as e:
            print(f"Error in BioClip list prediction: {e}")
            return []
