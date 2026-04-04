import os
import json
import copy
import random
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from .common import BaseDataset

# Data augmentation prompts for scene layout generation
CONSTRAINT_PROMPTS = {
    # Object scope constraints
    "object_scope": [
        "Only include the objects described above.",
        "The room contains only the mentioned furniture.", 
        "No extra furniture is included.",
        "Exclude any objects not listed.",
        "Do not add items beyond the description.",
        "Keep only the specified objects in the scene.",
        "The layout includes solely the mentioned items.",
        "Avoid adding any additional furniture pieces."
    ],
    
    # Small objects and details control
    "small_objects": [
        "Do not include small decorative items.",
        "No clutter or small objects.",
        "Avoid filling the room with tiny details.",
        "Keep the scene free of small decorations.",
        "Exclude accessories and ornaments.",
        "No small decorative pieces should be added.",
        "Avoid cluttering with minor objects.",
        "Keep decorative items to a minimum."
    ],
    
    # Simplicity and minimalism
    "minimal": [
        "The layout is simple and minimal.",
        "The room looks clean and uncluttered.",
        "The arrangement is minimalistic.",
        "The scene is spacious and not crowded.",
        "The room contains few objects.",
        "Keep the design simple and clean.",
        "Maintain a minimalist aesthetic.",
        "The space should feel open and airy."
    ],
    
    # Rich details and complexity
    "complex": [
        "The scene includes many detailed elements.",
        "Add rich details to the objects.",
        "The layout is visually complex.",
        "Many small items are neatly placed.",
        "The room is decorated with lots of details.",
        "Include abundant decorative elements.",
        "The space is richly furnished.",
        "Add intricate details throughout the scene."
    ],
    
    # Arrangement and organization
    "organized": [
        "Objects are neatly arranged.",
        "Furniture is aligned in an orderly manner.",
        "Items are placed symmetrically.",
        "The room looks tidy and organized.",
        "Objects are positioned with clear spacing.",
        "Everything is perfectly aligned.",
        "The arrangement follows a systematic pattern.",
        "Items are carefully positioned for balance."
    ],
    
    # Natural and realistic placement
    "realistic": [
        "Objects are placed naturally.",
        "The arrangement looks lived-in.",
        "Items are positioned realistically.",
        "The layout feels authentic.",
        "Objects follow natural placement patterns.",
        "The scene has a realistic flow.",
        "Items are arranged as people would use them.",
        "The placement feels organic and natural."
    ]
}

# Sentence rewriting templates for style normalization
SENTENCE_REWRITE_TEMPLATES = {
    # Template 1: Simple enumeration style (similar to 3D-Front)
    "simple": [
        "The room has {objects}.",
        "This space contains {objects}.",
        "The layout includes {objects}.",
        "There is {objects} in the room.",
        "The room features {objects}."
    ],
    
    # Template 2: Descriptive style (similar to MP3D)
    "descriptive": [
        "This is {room_type} that contains {objects}. {spatial_info}",
        "The space is {room_type} with {objects}. {spatial_info}",
        "This {room_type} includes {objects}. {spatial_info}",
        "The room serves as {room_type} featuring {objects}. {spatial_info}"
    ],
    
    # Template 3: Structured style (similar to Infinigen)
    "structured": [
        "The room is {room_type}. There is {objects}. {organization_info}",
        "This is {room_type}. The space contains {objects}. {organization_info}",
        "The space functions as {room_type}. It has {objects}. {organization_info}"
    ]
}

# Common spatial relationship phrases
SPATIAL_PHRASES = [
    "Items are arranged throughout the space.",
    "Objects are positioned strategically.",
    "The furniture is placed for optimal use.",
    "Elements are distributed across the room.",
    "The layout maximizes the available space."
]

# Organization information phrases
ORGANIZATION_PHRASES = [
    "The space is well organized.",
    "Everything is arranged systematically.",
    "The room has a functional layout.",
    "The arrangement is practical.",
    "The space is efficiently designed."
]

# NEW: Import text embedding utilities
try:
    from scene_synthesis.utils.text_embedding_utils import CategoryTextEmbedder
    TEXT_EMBEDDING_AVAILABLE = True
except ImportError:
    TEXT_EMBEDDING_AVAILABLE = False


class M3DLayoutScene:
    """Represents a single scene in M3DLayout format."""
    def __init__(self, scene_id: str, objects: List[Dict[str, Any]], description: Optional[Dict] = None, encoding_type: str = "diffusion", dataset_source: str = "unknown", large_object_categories: Optional[List[str]] = None):
        self.scene_id = scene_id
        self.objects = objects
        self.description = description or {}
        self.encoding_type = encoding_type
        self.dataset_source = dataset_source
        self.large_object_categories = large_object_categories or []
        
        # Compute scene graph for large objects
        self.scene_graph = self._compute_scene_graph()

    @property
    def class_labels(self):
        """Get class labels of all objects in the scene"""
        return [obj["category"] for obj in self.objects]

    @property
    def translations(self):
        """Get positions of all objects in the scene"""
        return np.array([obj["location"] for obj in self.objects])

    @property
    def sizes(self):
        """Get sizes of all objects in the scene"""
        return np.array([obj["size"] for obj in self.objects])

    @property
    def angles(self):
        """Get angles of all objects in the scene in cos/sin format for better periodicity handling"""
        angles_rad = np.array([obj["rotation"] for obj in self.objects])
        # Convert to cos/sin representation: [cos(angle), sin(angle)]
        if self.encoding_type == "diffusion":
            cos_angles = np.cos(angles_rad).reshape(-1, 1)
            sin_angles = np.sin(angles_rad).reshape(-1, 1)
            return np.concatenate([cos_angles, sin_angles], axis=1)
        elif self.encoding_type == "autoregressive":
            return angles_rad.reshape(-1, 1)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
    def __len__(self):
        return len(self.objects)

    def _compute_scene_graph(self) -> Dict[str, Any]:
        """
        Compute scene graph from large objects' layout.
        Returns a dictionary containing nodes and edges representing spatial relationships.
        """
        if not self.large_object_categories:
            return {"nodes": [], "edges": []}
        
        # Filter large objects
        large_objects = []
        for i, obj in enumerate(self.objects):
            if obj["category"] in self.large_object_categories:
                large_objects.append((i, obj))  # Keep original index
        
        if len(large_objects) < 2:
            return {"nodes": [], "edges": []}
        
        nodes = []
        edges = []
        
        # Create nodes
        for idx, obj in large_objects:
            nodes.append({
                "object_idx": idx,
                "category": obj["category"],
                "position": obj["location"],
                "size": obj["size"]
            })
        
        # Create edges (relationships between objects)
        for i in range(len(large_objects)):
            for j in range(i + 1, len(large_objects)):
                idx1, obj1 = large_objects[i]
                idx2, obj2 = large_objects[j]
                
                # Compute spatial relationships
                relationships = self._compute_spatial_relationships(obj1, obj2)
                
                if relationships:  # Only add edge if there are relationships
                    # Get category indices from the class order
                    # Load class_order from train_stats_file if available
                    class_order = {}
                    if hasattr(self, 'config') and 'train_stats_file' in self.config:
                        try:
                            import json
                            with open(self.config['train_stats_file'], 'r') as f:
                                stats = json.load(f)
                                class_order = stats.get('class_order', {})
                        except:
                            pass
                    
                    cat1_idx = class_order.get(obj1["category"], -1)
                    cat2_idx = class_order.get(obj2["category"], -1)
                    
                    edges.append({
                        "node1_idx": i,  # Index in nodes list
                        "node2_idx": j,
                        "object1_idx": idx1,  # Original object index
                        "object2_idx": idx2,
                        "subject_class_idx": cat1_idx,  # Category index for subject
                        "object_class_idx": cat2_idx,   # Category index for object
                        "relationships": relationships
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "num_large_objects": len(large_objects)
        }
    
    def _compute_spatial_relationships(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> List[str]:
        """
        Compute spatial relationships between two objects.
        Returns a list of relationship types.
        """
        pos1 = np.array(obj1["location"])
        pos2 = np.array(obj2["location"])
        size1 = np.array(obj1["size"])
        size2 = np.array(obj2["size"])
        
        relationships = []
        
        # Compute differences
        diff = pos2 - pos1
        distance = np.linalg.norm(diff)
        
        # Normalize differences (rough direction)
        if distance > 1e-6:
            dir_vector = diff / distance
        else:
            dir_vector = np.array([0, 0, 0])
        
        # Left/Right relationships (based on x-axis)
        if abs(dir_vector[0]) > 0.3:  # Significant x-component
            if dir_vector[0] > 0:
                relationships.append("right_of")  # obj2 is right of obj1
            else:
                relationships.append("left_of")   # obj2 is left of obj1
        
        # Front/Back relationships (based on y-axis)
        if abs(dir_vector[1]) > 0.3:  # Significant y-component
            if dir_vector[1] > 0:
                relationships.append("front_of")  # obj2 is front of obj1
            else:
                relationships.append("back_of")   # obj2 is back of obj1
        
        # Above/Below relationships (based on z-axis)
        height_diff = pos2[2] - pos1[2]
        if abs(height_diff) > 0.1:  # Significant height difference
            if height_diff > 0:
                relationships.append("below")  # obj2 is below obj1 (higher z)
            else:
                relationships.append("above")  # obj2 is above obj1 (lower z)
        
        # Proximity relationships
        avg_size = (np.mean(size1) + np.mean(size2)) / 2
        if distance < avg_size * 2:
            relationships.append("close_to")
        elif distance > avg_size * 5:
            relationships.append("far_from")
        
        return relationships
        

class M3DLayoutDataset(BaseDataset):
    """M3DLayout dataset class."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize M3DLayoutDataset
        
        Args:
            config: Configuration dictionary containing json_files and other settings
        """
        self.config = config
        self.json_files = config.get("json_files", [])
        self.encoding_type = config.get("encoding_type", "diffusion")
        self.json_files = [f for f in self.json_files if os.path.exists(f)]
        if not self.json_files:
            raise FileNotFoundError("No M3DLayout JSON files found")
        
        # Object types (categories) in the dataset
        self._object_types = config.get("object_categories", None)  # If None, will be determined from data

        # Text embedding configuration
        self.use_text_embedding_for_categories = config.get("use_text_embedding_for_categories", False)
        self.category_text_embed_dim = config.get("category_text_embed_dim", 512)
        self.category_text_embedding_type = config.get("category_text_embedding_type", "bert")
        
        # Two-stage training configuration
        self.two_stage_small_from_large = config.get("two_stage_small_from_large", False)
        self.large_object_categories = config.get("large_object_categories", [])
        
        # Data augmentation configuration
        self.enable_description_augmentation = config.get("enable_description_augmentation", True)
        self.augmentation_probability = config.get("augmentation_probability", 0.8)
        
        # Sentence rewriting for style normalization
        self.enable_sentence_rewriting = config.get("enable_sentence_rewriting", True)
        self.sentence_rewriting_probability = config.get("sentence_rewriting_probability", 0.6)
        
        # Small object filtering for Infinigen data
        self.enable_small_object_filtering = config.get("enable_small_object_filtering", True)
        self.small_object_filtering_probability = config.get("small_object_filtering_probability", 0.5)
        
        self._load_data()
        self._compute_properties()
        self._apply_scaling()
        
        # Compute small object categories as difference between all object types and large object categories
        self._compute_small_object_categories()
        
        # NEW: Initialize text embedder if needed
        if self.use_text_embedding_for_categories:
            self._initialize_text_embedder()
        
        self.save_dataset_stats(self.config["train_stats_file"])

    def _detect_dataset_source(self, scene_id: str, json_file_path: str) -> str:
        """
        Detect which dataset a scene comes from based on scene_id and file path
        
        Args:
            scene_id: Scene identifier
            json_file_path: Path to the JSON file containing the scene
            
        Returns:
            Dataset source: '3dfront', 'infinigen', 'mp3d', or 'unknown'
        """
        # Check file path for dataset indicators
        file_path_lower = json_file_path.lower()
        
        if '3dfront' in file_path_lower or 'threedfront' in file_path_lower:
            return '3dfront'
        elif 'infinigen' in file_path_lower:
            return 'infinigen'
        elif 'mp3d' in file_path_lower or 'matterport' in file_path_lower:
            return 'mp3d'
        else:
            raise ValueError(f"Unable to determine dataset source from file path: {json_file_path}")

    def _compute_small_object_categories(self):
        """
        Compute small object categories as the difference between all object types and large object categories
        """
        # Convert large_object_categories to set for efficient operations
        large_categories_set = set(self.large_object_categories)
        
        # Compute small object categories as difference
        self.small_object_categories = [
            category for category in self._object_types 
            if category not in large_categories_set
        ]
        
        print(f"Computed {len(self.small_object_categories)} small object categories: {self.small_object_categories}")
        print(f"Large object categories: {len(self.large_object_categories)} items")

    def _filter_small_objects(self, scene: 'M3DLayoutScene') -> tuple['M3DLayoutScene', bool]:
        """
        Filter small objects from Infinigen scenes with 50% probability
        
        Args:
            scene: Original scene object
            
        Returns:
            Tuple of (filtered_scene, was_filtered)
        """
        # Only apply to Infinigen dataset
        if scene.dataset_source != 'infinigen':
            return scene, False
            
        # Skip if filtering is disabled
        if not self.enable_small_object_filtering:
            return scene, False
            
        # Apply filtering with specified probability (default 50%)
        if random.random() > self.small_object_filtering_probability:
            return scene, False
        
        # Filter out small objects
        filtered_objects = []
        for obj in scene.objects:
            if obj["category"] not in self.small_object_categories:
                filtered_objects.append(obj)
        
        # Ensure we still have at least 2 objects after filtering
        if len(filtered_objects) < 2:
            return scene, False  # Don't filter if it would result in too few objects
        
        # Create new scene with filtered objects
        filtered_scene = M3DLayoutScene(
            scene_id=scene.scene_id,
            objects=filtered_objects,
            description=scene.description,
            encoding_type=scene.encoding_type,
            dataset_source=scene.dataset_source
        )
        
        return filtered_scene, True

    def _extract_objects_from_description(self, description: str) -> list:
        """
        Extract object mentions from description text using improved patterns
        
        Args:
            description: Original description text
            
        Returns:
            List of extracted object names
        """
        import re
        
        # Extended furniture vocabulary from the actual object types
        known_furniture = {
            # Basic furniture
            'bed', 'sofa', 'chair', 'table', 'desk', 'cabinet', 'shelf', 'bookshelf',
            'wardrobe', 'nightstand', 'side_table', 'coffee_table', 'dining_table',
            'dining_chair', 'armchair', 'loveseat', 'multi_seat_sofa', 
            
            # Compound names (common variations)
            'corner_side_table', 'loveseat_sofa', 'multi_seat_sofa', 'shelving_unit',
            'shelving_units', 'side_table', 'coffee_table', 'dining_chair',
            
            # Bathroom items
            'toilet', 'bathtub', 'sink', 'vanity', 'towel', 'curtain',
            
            # Other items
            'tv', 'tv_stand', 'lamp', 'floor_lamp', 'desk_lamp', 'plant', 'planter',
            'vase', 'rug', 'cushion', 'pillow', 'window', 'door', 'mirror',
            'microwave', 'oven', 'dishwasher', 'fan', 'trash_can', 'trashcan',
            'bench', 'stool', 'column', 'picture', 'artwork', 'couch'
        }
        
        # Alternative names mapping
        furniture_aliases = {
            'couch': 'sofa',
            'trashcan': 'trash_can',
            'black_couch': 'sofa',
            'black_couch': 'sofa'
        }
        
        extracted_objects = set()
        description_lower = description.lower()
        
        # Method 1: Direct word matching (most reliable)
        words = re.findall(r'\b\w+\b', description_lower)
        for i, word in enumerate(words):
            # Check single words
            if word in known_furniture:
                extracted_objects.add(word)
            
            # Check two-word combinations
            if i < len(words) - 1:
                two_word = f"{word}_{words[i+1]}"
                if two_word in known_furniture:
                    extracted_objects.add(two_word)
            
            # Check three-word combinations for specific cases
            if i < len(words) - 2:
                three_word = f"{word}_{words[i+1]}_{words[i+2]}"
                if three_word in known_furniture:
                    extracted_objects.add(three_word)
        
        # Method 2: Pattern-based extraction with short phrases only
        simple_patterns = [
            r'\b(?:a|an|the)\s+((?:[a-z]+\s+){0,2}[a-z]+)\b',  # "a table", "a coffee table"
            r'\bcontains?\s+(?:a|an|the)?\s*((?:[a-z]+\s+){0,2}[a-z]+)\b',  # "contains a table"
            r'\bthere\s+(?:is|are)\s+(?:a|an|the)?\s*((?:[a-z]+\s+){0,2}[a-z]+)\b'  # "there is a table"
        ]
        
        for pattern in simple_patterns:
            matches = re.findall(pattern, description_lower)
            for match in matches:
                # Clean up and normalize
                obj = match.strip().replace(' ', '_')
                
                # Only keep known furniture
                if obj in known_furniture:
                    extracted_objects.add(obj)
                # Check for aliases
                elif obj in furniture_aliases:
                    extracted_objects.add(furniture_aliases[obj])
                # Check if it's a partial match of known furniture
                else:
                    for furniture in known_furniture:
                        if furniture in obj and len(furniture) > 3:
                            extracted_objects.add(furniture)
                            break
        
        # Convert to list and clean up
        result = []
        for obj in extracted_objects:
            # Apply aliases
            if obj in furniture_aliases:
                obj = furniture_aliases[obj]
            
            # Only add if it's a known furniture item
            if obj in known_furniture and obj not in result:
                result.append(obj)
        
        return result[:6]  # Limit to 6 objects for practical reasons

    def _infer_room_type(self, description: str, objects: list) -> str:
        """
        Infer room type from description and objects with better accuracy
        
        Args:
            description: Original description
            objects: List of objects in the scene
            
        Returns:
            Inferred room type
        """
        description_lower = description.lower()
        
        # Enhanced room type detection
        room_keywords = {
            'bedroom': ['bedroom', 'bed', 'nightstand', 'wardrobe'],
            'living room': ['living', 'sofa', 'coffee_table', 'tv', 'loveseat', 'armchair'],
            'dining room': ['dining', 'dining_table', 'dining_chair'],
            'kitchen': ['kitchen', 'sink', 'oven', 'microwave', 'dishwasher'],
            'bathroom': ['bathroom', 'toilet', 'bathtub', 'sink', 'vanity', 'towel'],
            'office': ['office', 'desk', 'desk_chair'],
            'porch': ['porch', 'garden', 'outdoor', 'veranda', 'patio'],
            'entryway': ['entryway', 'foyer', 'lobby', 'entrance'],
            'outdoor space': ['outdoor', 'patio', 'veranda', 'garden']
        }
        
        # Check description for explicit room type mentions
        for room_type, keywords in room_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return f"a {room_type}"
        
        # Check objects for room type clues
        object_string = ' '.join(objects).lower()
        for room_type, keywords in room_keywords.items():
            if any(keyword in object_string for keyword in keywords):
                return f"a {room_type}"
        
        return "a room"

    def _extract_spatial_info(self, description: str) -> str:
        """
        Extract spatial relationship information from description
        
        Args:
            description: Original description text
            
        Returns:
            Spatial information string
        """
        import re
        
        spatial_patterns = [
            r'[^.]*(?:positioned|placed|located|situated)[^.]*\.',
            r'[^.]*(?:next to|against|near|in front of|behind|opposite|adjacent to)[^.]*\.',
            r'[^.]*(?:in the center|centrally|along the wall|against.*wall)[^.]*\.',
            r'[^.]*(?:upper part|lower part|corner|side)[^.]*\.',
        ]
        
        spatial_info = []
        for pattern in spatial_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            spatial_info.extend(matches)
        
        if spatial_info:
            return ' '.join(spatial_info[:2])  # Limit to 2 spatial descriptions
        
        return random.choice(SPATIAL_PHRASES)

    def _rewrite_sentence(self, description: str, scene_objects: list) -> str:
        """
        Rewrite sentence to normalize style across different datasets with better preservation of information
        
        Args:
            description: Original description text
            scene_objects: List of object categories in the scene
            
        Returns:
            Rewritten description with normalized style
        """
        if not self.enable_sentence_rewriting:
            return description
            
        # Apply rewriting with specified probability
        if random.random() > self.sentence_rewriting_probability:
            return description
        
        # Extract information from original description
        extracted_objects = self._extract_objects_from_description(description)
        room_type = self._infer_room_type(description, extracted_objects + scene_objects)
        spatial_info = self._extract_spatial_info(description)
        
        # Use scene objects if extraction failed, but clean them up
        if not extracted_objects:
            # Clean up scene object names
            cleaned_objects = []
            for obj in scene_objects[:6]:  # Limit to first 6 objects
                # Convert underscores to spaces and handle common variations
                clean_obj = obj.replace('_', ' ')
                if clean_obj not in cleaned_objects:
                    cleaned_objects.append(clean_obj)
            extracted_objects = cleaned_objects
        
        # Format object list more naturally
        if not extracted_objects:
            objects_text = "furniture"
        elif len(extracted_objects) == 1:
            objects_text = f"a {extracted_objects[0].replace('_', ' ')}"
        elif len(extracted_objects) == 2:
            objects_text = f"a {extracted_objects[0].replace('_', ' ')} and a {extracted_objects[1].replace('_', ' ')}"
        else:
            formatted_objects = [obj.replace('_', ' ') for obj in extracted_objects[:5]]  # Limit to 5 objects
            if len(formatted_objects) > 3:
                # For many objects, use more natural phrasing
                objects_text = f"furniture including a {formatted_objects[0]}, a {formatted_objects[1]}, a {formatted_objects[2]}, and other items"
            else:
                objects_text = f"a {', a '.join(formatted_objects[:-1])}, and a {formatted_objects[-1]}"
        
        # Select rewriting style with consideration of original length and complexity
        original_length = len(description.split())
        
        if original_length > 25:  # Long descriptions get descriptive style
            template_styles = ['descriptive', 'structured']
        elif original_length < 10:  # Short descriptions get simple style
            template_styles = ['simple']
        else:  # Medium descriptions get mixed styles
            template_styles = ['simple', 'structured']
        
        selected_style = random.choice(template_styles)
        
        # Apply the selected template with better variety
        if selected_style == 'simple':
            templates = [
                "The room has {objects}.",
                "This space contains {objects}.", 
                "The layout includes {objects}.",
                "There is {objects} in the room."
            ]
            template = random.choice(templates)
            rewritten = template.format(objects=objects_text)
            
        elif selected_style == 'descriptive':
            templates = [
                "This is {room_type} that contains {objects}. {spatial_info}",
                "The space is {room_type} with {objects}. {spatial_info}",
                "This {room_type} features {objects}. {spatial_info}",
            ]
            template = random.choice(templates)
            rewritten = template.format(
                room_type=room_type,
                objects=objects_text,
                spatial_info=spatial_info
            )
            
        else:  # structured
            templates = [
                "The room is {room_type}. There is {objects}. {organization_info}",
                "This is {room_type}. The space contains {objects}. {organization_info}",
                "The space functions as {room_type}. It has {objects}. {organization_info}"
            ]
            template = random.choice(templates)
            organization_info = random.choice(ORGANIZATION_PHRASES)
            rewritten = template.format(
                room_type=room_type,
                objects=objects_text,
                organization_info=organization_info
            )
        
        return rewritten

    def _augment_description(self, description: str, dataset_source: str, num_objects: int, small_objects_filtered: bool = False) -> str:
        """
        Augment scene description with constraint prompts based on dataset source and object count
        
        Args:
            description: Original description text
            dataset_source: Source dataset ('3dfront', 'infinigen', 'mp3d', 'unknown')
            num_objects: Number of objects in the scene
            small_objects_filtered: Whether small objects were filtered out (for Infinigen)
            
        Returns:
            Augmented description with added constraint prompts
        """
        if not self.enable_description_augmentation:
            return description

        # Check if we should apply augmentation (80% probability)
        if random.random() > self.augmentation_probability:
            return description
        
        # Select appropriate prompt categories based on dataset source
        selected_prompts = []
        
        if dataset_source == '3dfront':
            # 3D-Front: Simple and minimal descriptions
            prompt_categories = ['minimal', 'object_scope', 'small_objects', 'organized']
            
        elif dataset_source == 'infinigen':
            # Infinigen: Adjust prompts based on whether small objects were filtered
            if small_objects_filtered:
                # When small objects are filtered: use minimal and organized prompts
                prompt_categories = ['minimal', 'object_scope', 'small_objects', 'organized']
            else:
                # When small objects are preserved: use complex prompts
                prompt_categories = ['complex', 'realistic', 'organized']
            
        elif dataset_source == 'mp3d':
            # MP3D: Based on object count
            if num_objects <= 5:
                prompt_categories = ['minimal', 'object_scope', 'small_objects']
            elif num_objects <= 10:
                prompt_categories = ['organized', 'realistic']
            else:
                prompt_categories = ['complex', 'realistic']
                
        else:
            raise ValueError(f"Unknown dataset source: {dataset_source}")
        
        # Randomly select 1-2 prompt categories
        num_categories = random.randint(1, min(2, len(prompt_categories)))
        selected_categories = random.sample(prompt_categories, num_categories)
        
        # Select one prompt from each chosen category
        for category in selected_categories:
            if category in CONSTRAINT_PROMPTS:
                prompt = random.choice(CONSTRAINT_PROMPTS[category])
                selected_prompts.append(prompt)
        
        if not selected_prompts:
            return description
        
        # Randomly decide whether to add prompts at the beginning or end
        add_at_beginning = random.choice([True, False])
        
        # Combine prompts
        prompt_text = " ".join(selected_prompts)
        
        if add_at_beginning:
            augmented_description = f"{prompt_text} {description}".strip()
        else:
            augmented_description = f"{description} {prompt_text}".strip()
        
        return augmented_description

    def _initialize_text_embedder(self):
        """Initialize text embedder for category names"""
        if not TEXT_EMBEDDING_AVAILABLE:
            raise ImportError("Text embedding utilities not available. Please install required packages.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.text_embedder = CategoryTextEmbedder(
                embedding_type=self.category_text_embedding_type,
                embed_dim=self.category_text_embed_dim,
                device=device
            )
            
            # Pre-compute embeddings for all categories
            self.category_embeddings = self.text_embedder.create_category_embedding_dict(self._object_types)
            print(f"Initialized text embedder with {self.category_text_embedding_type} embeddings")
            
        except Exception as e:
            print(f"Warning: Failed to initialize text embedder: {e}")
            print("Falling back to one-hot encoding")
            self.use_text_embedding_for_categories = False

    def _load_data(self):
        """Load all JSON files"""
        self.scenes = []
        all_categories = set()
        filter_count = 0
        
        for json_file in self.json_files:
            print(f"Loading: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            all_categories.update(data["categories"])
            
            for scene_data in data["scenes"]:

                if len(scene_data["objects"]) < 2:  # filter scenes with less than 2 objects
                    filter_count += 1
                    continue
                
                # Detect dataset source
                dataset_source = self._detect_dataset_source(scene_data["scene_id"], json_file)
                
                scene = M3DLayoutScene(
                    scene_id=scene_data["scene_id"],
                    objects=scene_data["objects"],
                    description=scene_data.get("description", {}),
                    encoding_type=self.encoding_type,
                    dataset_source=dataset_source,
                    large_object_categories=self.large_object_categories
                )
                
                self.scenes.append(scene)
        
        print(f"Total loaded {len(self.scenes)} scenes, filtered {filter_count} scenes with less than 2 objects")

        if self._object_types is None:
            self._object_types = sorted(list(all_categories))
            print(f"Found {len(self._object_types)} object types: {self._object_types}")
        else:
            print(f"Found {len(list(all_categories))} object types, but using predefined {len(self._object_types)} types: {self._object_types}")

    def _compute_properties(self):
        """Compute dataset statistics"""
        if not self.scenes:
            return
        
        # Compute bounds
        all_translations = []
        all_sizes = []
        all_angles = []
        
        # Compute category frequencies
        category_counts = {}
        
        for scene in self.scenes:
            all_translations.extend(scene.translations.tolist())
            all_sizes.extend(scene.sizes.tolist())
            all_angles.extend(scene.angles.tolist())
            
            for category in scene.class_labels:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Compute bounds
        if all_translations:
            all_translations = np.array(all_translations)
            all_sizes = np.array(all_sizes)
            all_angles = np.array(all_angles)  # shape: (total_objects, 2) for [cos, sin]
            
            self._centroids = (np.min(all_translations, axis=0), np.max(all_translations, axis=0))
            self._sizes = (np.min(all_sizes, axis=0), np.max(all_sizes, axis=0))
            # For cos/sin angles, the range is naturally [-1, 1], but we compute actual bounds
            self._angles = (np.min(all_angles, axis=0), np.max(all_angles, axis=0))
        else:
            self._centroids = (np.array([-5.0, -5.0, 0.0]), np.array([5.0, 5.0, 3.0]))
            self._sizes = (np.array([0.1, 0.1, 0.1]), np.array([3.0, 3.0, 3.0]))
            # Default bounds for cos/sin angles
            self._angles = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        
        # Set category-related properties
        self._class_labels = self._object_types + ["start", "end"]
        self._class_frequencies = {k: v / sum(category_counts.values()) for k, v in category_counts.items()}
        self._class_order = {k: i for i, k in enumerate(category_counts.keys())}
        self._count_furniture = category_counts
        
        # Set max length
        self._max_length = self.config.get("sample_num_points", 270)

    def _apply_scaling(self):
        """Apply scaling"""
        # compute scaling parameters
        t_min, t_max = self._centroids
        s_min, s_max = self._sizes
        a_min, a_max = self._angles
        
        # store scaling parameters
        self._translation_scale = (t_max - t_min) / 2.0
        self._translation_offset = (t_max + t_min) / 2.0
        self._size_scale = (s_max - s_min) / 2.0
        self._size_offset = (s_max + s_min) / 2.0
        self._angle_scale = (a_max - a_min) / 2.0
        self._angle_offset = (a_max + a_min) / 2.0

        # print scaling parameters
        print(f"Translation scale: {self._translation_scale}, Translation offset: {self._translation_offset}")
        print(f"Size scale: {self._size_scale}, Size offset: {self._size_offset}")
        print(f"Angle scale: {self._angle_scale}, Angle offset: {self._angle_offset}")

    def _get_2d_polygon(self, translation, size, angle):
        """
        Compute 2D polygon (top-down view) for an object given its translation, size, and angle.
        
        Args:
            translation: [x, y, z] position (center coordinates)
            size: [half_width, half_depth, half_height] (half extents)
            angle: rotation angle in radians (for autoregressive) or [cos, sin] (for diffusion)
            
        Returns:
            Shapely Polygon object representing the 2D footprint
        """
        # Extract x, z position for top-down view (xz plane)
        x, z = translation[0], translation[2]
        
        # Size contains half extents, so full dimensions are 2 * size
        half_w, half_d = size[0], size[2]
        
        # Handle angle format
        if isinstance(angle, (list, np.ndarray)) and len(angle) == 2:
            # Diffusion format: [cos, sin]
            cos_a, sin_a = angle
            theta = np.arctan2(sin_a, cos_a)
        else:
            # Autoregressive format: direct angle
            theta = float(angle)
        
        # Corner points before rotation (centered at origin)
        corners = np.array([
            [-half_w, -half_d],  # bottom-left
            [half_w, -half_d],   # bottom-right  
            [half_w, half_d],    # top-right
            [-half_w, half_d]    # top-left
        ])
        
        # Rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ])
        
        # Rotate corners
        rotated_corners = corners @ rotation_matrix.T
        
        # Translate to final position
        final_corners = rotated_corners + np.array([x, z])
        
        # Create polygon
        return Polygon(final_corners)

    def _compute_small_object_placements(self, translations, sizes, angles, class_labels):
        """
        Compute which large object each small object is placed on/in based on 2D polygon containment.
        
        Args:
            translations: Normalized translations array [N, 3]
            sizes: Normalized sizes array [N, 3] 
            angles: Normalized angles array [N, 1] or [N, 2]
            class_labels: List of category names [N]
            
        Returns:
            Dictionary mapping small object indices to containing large object indices
        """
        # Denormalize data for polygon computation
        denorm_translations, denorm_sizes, denorm_angles = self._denormalize_data(translations, sizes, angles)
        
        # Separate large and small objects
        large_indices = []
        small_indices = []
        for i, category in enumerate(class_labels):
            if category in self.large_object_categories:
                large_indices.append(i)
            else:
                small_indices.append(i)
        
        # Compute polygons for all objects
        large_polygons = []
        for idx in large_indices:
            poly = self._get_2d_polygon(denorm_translations[idx], denorm_sizes[idx], denorm_angles[idx])
            large_polygons.append((idx, poly))
        
        # Find containment relationships
        containment_map = {}
        for small_idx in small_indices:
            small_poly = self._get_2d_polygon(denorm_translations[small_idx], denorm_sizes[small_idx], denorm_angles[small_idx])
            
            # Check containment in each large object
            contained_in = []
            for large_idx, large_poly in large_polygons:
                try:
                    if large_poly.contains(small_poly) or large_poly.intersects(small_poly):
                        contained_in.append(large_idx)
                except:
                    continue  # Skip invalid geometries
            
            # If contained in multiple, pick the first one (could be improved with area overlap)
            if contained_in:
                containment_map[small_idx] = contained_in[0]
        
        return containment_map

    def _visualize_containment(self, scene_id, translations, sizes, angles, class_labels, containment_map, save_path=None):
        """
        Visualize the containment relationships between small and large objects.
        
        Args:
            scene_id: Scene identifier for filename
            translations: Denormalized translations
            sizes: Denormalized sizes
            angles: Denormalized angles
            class_labels: Class labels
            containment_map: Mapping from small object indices to large object indices
            save_path: Path to save visualization (optional)
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot all objects
        large_patches = []
        small_patches = []
        
        for i in range(len(translations)):
            poly = self._get_2d_polygon(translations[i], sizes[i], angles[i])
            
            # Get category name
            category_name = class_labels[i]
            
            # Convert polygon to matplotlib patch
            x, y = poly.exterior.xy
            if category_name in self.large_object_categories:
                patch = plt.Polygon(np.column_stack([x, y]), facecolor='lightblue', edgecolor='blue', alpha=0.7, label='Large Objects')
                large_patches.append(patch)
            else:
                if i in containment_map:
                    patch = plt.Polygon(np.column_stack([x, y]), facecolor='lightcoral', edgecolor='red', alpha=0.7, label='Contained Small Objects')
                else:
                    patch = plt.Polygon(np.column_stack([x, y]), facecolor='lightgreen', edgecolor='green', alpha=0.7, label='Free Small Objects')
                small_patches.append(patch)
            
            ax.add_patch(patch)
            
            # Add category label
            centroid = poly.centroid
            ax.text(centroid.x, centroid.y, category_name, ha='center', va='center', fontsize=8)
        
        # Draw containment lines
        for small_idx, large_idx in containment_map.items():
            small_poly = self._get_2d_polygon(translations[small_idx], sizes[small_idx], angles[small_idx])
            large_poly = self._get_2d_polygon(translations[large_idx], sizes[large_idx], angles[large_idx])
            
            small_centroid = small_poly.centroid
            large_centroid = large_poly.centroid
            
            ax.plot([small_centroid.x, large_centroid.x], [small_centroid.y, large_centroid.y], 
                   'k--', alpha=0.5, linewidth=1)
        
        ax.set_aspect('equal')
        ax.set_title(f'Scene {scene_id} - Object Containment Visualization')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Z coordinate')
        
        # Create legend
        handles = []
        if large_patches:
            handles.append(plt.Rectangle((0,0),1,1, facecolor='lightblue', edgecolor='blue', alpha=0.7, label='Large Objects'))
        if small_patches:
            handles.append(plt.Rectangle((0,0),1,1, facecolor='lightcoral', edgecolor='red', alpha=0.7, label='Contained Small Objects'))
            handles.append(plt.Rectangle((0,0),1,1, facecolor='lightgreen', edgecolor='green', alpha=0.7, label='Free Small Objects'))
        
        if handles:
            ax.legend(handles=handles)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.close()

    def _normalize_data(self, translations, sizes, angles):
        """Normalize data to [-1, 1] range"""
        # normalize coordinates
        normalized_translations = (translations - self._translation_offset) / (self._translation_scale + 1e-8)
        
        # normalize sizes
        normalized_sizes = (sizes - self._size_offset) / (self._size_scale + 1e-8)
        
        # normalize angles
        normalized_angles = (angles - self._angle_offset) / (self._angle_scale + 1e-8)
        
        return normalized_translations, normalized_sizes, normalized_angles

    def _denormalize_data(self, translations, sizes, angles):
        """Denormalize data from [-1, 1] range back to original range"""
        # denormalize coordinates
        denormalized_translations = translations * (self._translation_scale + 1e-8) + self._translation_offset
        
        # denormalize sizes
        denormalized_sizes = sizes * (self._size_scale + 1e-8) + self._size_offset
        
        # denormalize angles
        denormalized_angles = angles * (self._angle_scale + 1e-8) + self._angle_offset
        
        return denormalized_translations, denormalized_sizes, denormalized_angles

    def post_process(self, sample_params):
        """
        Post-process generated samples by denormalizing and converting back to original format.
        
        Args:
            sample_params: Dictionary containing the generated samples with keys:
                - translations: normalized translation values
                - sizes: normalized size values 
                - angles: normalized cos/sin angle values
                - class_labels: class labels (unchanged)
                - room_layout: room layout (unchanged)
                - description: description (unchanged)
                
        Returns:
            Dictionary with denormalized values and angles converted back to radians
        """
        processed_params = {}
        
        for k, v in sample_params.items():
            if k == "class_labels" or k == "room_layout" or k == "description":
                # Keep these unchanged
                processed_params[k] = v
            
            elif k == "translations":
                # Denormalize translations
                processed_params[k] = self._denormalize_translations(v)
                
            elif k == "sizes":
                # Denormalize sizes
                processed_params[k] = self._denormalize_sizes(v)
                
            elif k == "angles":
                # Convert cos/sin angles back to radians and denormalize
                processed_params[k] = self._denormalize_angles(v)
                
            else:
                # For any other keys, keep them unchanged
                processed_params[k] = v
        
        return processed_params

    def _denormalize_translations(self, normalized_translations):
        """Denormalize translation values"""
        return normalized_translations * (self._translation_scale + 1e-8) + self._translation_offset

    def _denormalize_sizes(self, normalized_sizes):
        """Denormalize size values"""
        return normalized_sizes * (self._size_scale + 1e-8) + self._size_offset

    def _denormalize_angles(self, normalized_angles):
        """Convert angles back to radians and denormalize"""
        
        # Check the shape to determine if it's cos/sin format (diffusion) or direct angles (autoregressive)
        if normalized_angles.shape[-1] == 2:
            # Diffusion format: cos/sin angles (2D)
            # First denormalize the cos/sin values
            denormalized_cos_sin = normalized_angles * (self._angle_scale + 1e-8) + self._angle_offset
            
            # Convert cos/sin back to angle in radians
            # angles = arctan2(sin, cos)
            if len(denormalized_cos_sin.shape) == 3:
                # Shape: (batch_size, num_objects, 2)
                angles = np.arctan2(denormalized_cos_sin[:, :, 1:2], denormalized_cos_sin[:, :, 0:1])
            else:
                # Shape: (num_objects, 2)
                angles = np.arctan2(denormalized_cos_sin[:, 1:2], denormalized_cos_sin[:, 0:1])
            
        elif normalized_angles.shape[-1] == 1:
            # Autoregressive format: direct angles (1D)
            # Simply denormalize the angle values
            angles = normalized_angles * (self._angle_scale[0] + 1e-8) + self._angle_offset[0]
            
        else:
            raise ValueError(f"Unexpected angle dimension: {normalized_angles.shape[-1]}. Expected 1 (autoregressive) or 2 (diffusion).")
        
        return angles

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        """Get encoded scene data"""
        scene = self.scenes[idx]
        
        # Apply small object filtering for Infinigen scenes (NEW)
        filtered_scene, small_objects_filtered = self._filter_small_objects(scene)
        scene = filtered_scene  # Use the filtered scene for processing
        
        # get scene attributes
        translations = scene.translations.astype(np.float32)
        sizes = scene.sizes.astype(np.float32)
        angles = scene.angles.astype(np.float32)
        
        # Use one-hot encoding for class labels
        class_labels = np.zeros((len(scene.objects), len(self._class_labels)), dtype=np.float32)
        for i, category in enumerate(scene.class_labels):
            if category in self._object_types:
                class_idx = self._object_types.index(category)
                class_labels[i, class_idx] = 1.0
    
        # # if permutation is enabled, permute the data
        # if self._enable_permutation:
        #     indices = np.random.permutation(len(scene.objects))
        #     class_labels = class_labels[indices]
        #     translations = translations[indices]
        #     sizes = sizes[indices]
        #     angles = angles[indices]
        
        # For one-hot encoding, use the defined end indices

        if self.encoding_type == "diffusion":
            new_class_labels = np.concatenate([class_labels[:, :-2], class_labels[:, -1:]], axis=-1) #hstack
            L, C = new_class_labels.shape
        elif self.encoding_type == "autoregressive":
            new_class_labels = class_labels
            L, C = new_class_labels.shape
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # normalize data
        discretized_translations = np.round((translations / 0.5)).astype(int)
        discretized_translations = np.vstack([discretized_translations, np.tile(np.zeros(3)[None, :], [self._max_length - L, 1])]).astype(int)
        translations, sizes, angles = self._normalize_data(translations, sizes, angles)

        # Pad the end label in the end of each sequence, and convert the class labels to -1, 1
        end_label = np.eye(C)[-1]
        class_labels = np.vstack([
            new_class_labels, np.tile(end_label[None, :], [self._max_length - L, 1])
        ]).astype(np.float32)
        if self.encoding_type == "diffusion":
            class_labels = class_labels * 2.0 - 1.0 
        translations = np.vstack([translations, np.tile(np.zeros(3)[None, :], [self._max_length - L, 1])]).astype(np.float32)
        sizes = np.vstack([sizes, np.tile(np.zeros(3)[None, :], [self._max_length - L, 1])]).astype(np.float32)
        if self.encoding_type == "diffusion":
            angles = np.vstack([angles, np.tile(np.zeros(2)[None, :], [self._max_length - L, 1])]).astype(np.float32)
        elif self.encoding_type == "autoregressive":
            angles = np.vstack([angles, np.tile(np.zeros(1)[None, :], [self._max_length - L, 1])]).astype(np.float32)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # Reorder all objects together by x, z, y coordinates (no separation of large/small)
        # Find large and small objects for later use
        large_indices = []
        small_indices = []
        for i, category in enumerate(scene.class_labels):
            if category in self.large_object_categories:
                large_indices.append(i)
            else:
                small_indices.append(i)
        
        # Sort all objects together by [x, z, y] coordinates
        all_object_indices = list(range(L))  # All objects before padding
        if all_object_indices:
            all_coords = discretized_translations[all_object_indices][:, [0, 2, 1]]  # Extract x, z, y in that order
            # np.lexsort sorts by last row first, so we pass (y, z, x) to get x, z, y priority
            sort_order = np.lexsort((all_coords[:, 2], all_coords[:, 1], all_coords[:, 0]))
            sorted_indices = [all_object_indices[i] for i in sort_order]
        else:
            sorted_indices = []

        # Reorder the sequence: sorted objects followed by padding
        new_order = sorted_indices + np.arange(len(sorted_indices), len(class_labels)).tolist()

        # Reorder all arrays according to new_order
        translations = translations[new_order]
        sizes = sizes[new_order] 
        angles = angles[new_order]
        class_labels = class_labels[new_order]
        
        # Initialize dependency_labels after reordering
        dependency_labels = np.zeros(len(class_labels), dtype=np.int32)
        
        # Extract large objects' information (after sorting, need to find them in sorted order)
        num_large = len(large_indices)
        # Find positions of large objects in the sorted sequence
        large_positions_in_sorted = []
        for i, idx in enumerate(sorted_indices):
            if idx in large_indices:
                large_positions_in_sorted.append(i)
        
        # Extract large objects from sorted positions
        if large_positions_in_sorted:
            large_translations = translations[large_positions_in_sorted]
            large_sizes = sizes[large_positions_in_sorted]
            large_angles = angles[large_positions_in_sorted]
            large_classes = class_labels[large_positions_in_sorted]
            
            # Create partial input for large objects (normalized format)
            large_partial_input = np.concatenate([
                large_translations, large_sizes, large_angles, large_classes
            ], axis=-1).astype(np.float32)
        else:
            # No large objects found, create empty condition
            feature_dim = 3 + 3 + (2 if self.encoding_type == "diffusion" else 1) + class_labels.shape[-1]
            large_partial_input = np.zeros((0, feature_dim), dtype=np.float32)

        # build return data
        data = {
            "class_labels": class_labels,
            "translations": translations,
            "sizes": sizes,
            "angles": angles,
            "dependency_labels": dependency_labels,
            "length": len(scene.objects),
            "room_layout": np.ones((1, 64, 64), dtype=np.float32),  # virtual room layout
            "data_source": scene.dataset_source,
        }

        if self.encoding_type == "autoregressive":
            n_boxes = np.random.randint(0, L+1)
            for k, v in list(data.items()):
                if k in {"length", "room_layout", "data_source"}:
                    continue
                data[k + "_tr"] = v[n_boxes]
                data[k] = v[:n_boxes]
            data["length"] = n_boxes

        # Two-stage training: add large objects as partial condition
        if self.two_stage_small_from_large:
            if large_indices:
                data["large_partial_input"] = large_partial_input
                data["num_large_objects"] = num_large
            else:
                # No large objects found, create empty condition
                feature_dim = 3 + 3 + 2 + class_labels.shape[-1]  # trans + size + angle + class
                data["large_partial_input"] = np.zeros((0, feature_dim), dtype=np.float32)
                data["num_large_objects"] = 0
        
        # add description (if exists)
        if hasattr(scene, 'description'):
            data["description"] = scene.description

        if isinstance(data["description"], dict):
            # Collect all description sentences
            default_sentences = []
            all_sentences = []
            
            # Traverse all first-level keys
            for main_key in ["global_description", "large_objects_description", "small_objects_description"]:
                if small_objects_filtered and main_key == "small_objects_description":
                    continue  # Skip small object descriptions if they were filtered out
                if main_key in data["description"]:
                    # Traverse all second-level keys under each first-level key
                    for sub_key, sentences in data["description"][main_key].items():
                        if isinstance(sentences, list) and len(sentences) > 0:
                            if sub_key == "room_type" or sub_key == "summary":
                                default_sentences.extend([s.strip() for s in sentences if s.strip()])
                            else:
                                # Add all non-empty sentences to the total list
                                all_sentences.extend([s.strip() for s in sentences if s.strip()])
            
            # Randomly select 0-3 sentences
            if all_sentences or default_sentences:
                num_sentences = random.randint(0, min(3, len(all_sentences)))
                selected_sentences = random.sample(all_sentences, num_sentences)
                data["description"] = " ".join(default_sentences + selected_sentences)
            else:
                data["description"] = " ".join(default_sentences)
                if not data["description"]:
                    print(f"Scene {scene.scene_id} has no description")
        
        # Apply sentence rewriting for style normalization (before augmentation)
        if isinstance(data["description"], str) and data["description"].strip():
            # Apply sentence rewriting first
            data["description"] = self._rewrite_sentence(
                data["description"],
                scene.class_labels
            )
            
            # Then apply constraint augmentation
            data["description"] = self._augment_description(
                data["description"], 
                scene.dataset_source, 
                len(scene.objects),
                small_objects_filtered
            )
        elif not data.get("description"):
            # If no description exists, create a minimal one for processing
            data["description"] = "A room layout with furniture."
            
            # Apply sentence rewriting
            data["description"] = self._rewrite_sentence(
                data["description"],
                scene.class_labels
            )
            
            # Then apply constraint augmentation
            data["description"] = self._augment_description(
                data["description"], 
                scene.dataset_source, 
                len(scene.objects),
                small_objects_filtered
            )

        # print(f"Scene {scene.scene_id} from {scene.dataset_source} description: {data['description']}")
        
        # Add object size masks for timestep-staged training
        # Create masks indicating which objects are large (1) vs small (0)
        object_size_masks = np.zeros(len(class_labels), dtype=np.float32)
        
        # Mark large objects (which are now at the beginning due to reordering)
        if num_large > 0:
            object_size_masks[:num_large] = 1.0
        
        data["object_size_masks"] = object_size_masks
        
        # Add scene graph information
        data["scene_graph"] = scene.scene_graph
        
        # Compute small object containment relationships (AFTER data truncation for autoregressive)
        # Extract category names from current class_labels (after potential truncation)
        current_class_labels = []
        for i in range(len(data["class_labels"])):
            if len(data["class_labels"].shape) > 1:
                category_idx = np.argmax(data["class_labels"][i])
                if category_idx < len(self._object_types):
                    current_class_labels.append(self._object_types[category_idx])
                else:
                    current_class_labels.append("unknown")
            else:
                current_class_labels.append("unknown")
        
        containment_map = self._compute_small_object_placements(
            data["translations"],  # Use current (potentially truncated) data
            data["sizes"], 
            data["angles"], 
            current_class_labels  # Use current category names
        )
        # data["small_object_containment"] = containment_map
        
        # Compute dependency labels for each object
        # 0: large object or small object with no dependency
        # N+1: small object depends on object at index N in the sequence
        dependency_labels = np.zeros(self._max_length + 1, dtype=np.int32)
        
        # Mark dependencies based on containment_map
        for small_idx, large_idx in containment_map.items():
            if small_idx < len(dependency_labels):  # Ensure index is valid
                dependency_labels[small_idx] = large_idx + 1  # 1-based indexing for dependencies
        
        data["dependency_labels"] = dependency_labels
        
        # # Generate visualization for containment (save to file)
        # if containment_map:  # Only visualize if there are containment relationships
        #     # Denormalize data for visualization
        #     vis_translations, vis_sizes, vis_angles = self._denormalize_data(
        #         data["translations"], 
        #         data["sizes"], 
        #         data["angles"]
        #     )
            
        #     # Create visualization directory
        #     vis_dir = os.path.join("visualizations", "containment")
        #     os.makedirs(vis_dir, exist_ok=True)
            
        #     # Save visualization
        #     vis_path = os.path.join(vis_dir, f"{scene.scene_id}_containment.png")
        #     self._visualize_containment(
        #         scene.scene_id, 
        #         vis_translations, 
        #         vis_sizes, 
        #         vis_angles, 
        #         current_class_labels,  # Use current category names
        #         containment_map, 
        #         save_path=vis_path
        #     )
        
        return data

    def save_dataset_stats(self, output_path: str):
        """Save dataset statistics to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        dataset_stats = {
            "bounds_translations": self._centroids[0].tolist() + self._centroids[1].tolist(),
            "bounds_sizes": self._sizes[0].tolist() + self._sizes[1].tolist(),
            "bounds_angles": self._angles[0].tolist() + self._angles[1].tolist(),
            "class_labels": self._class_labels,
            "object_types": self._object_types,
            "class_frequencies": self._class_frequencies,
            "class_order": self._class_order,
            "count_furniture": self._count_furniture
        };
        
        with open(output_path, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        print(f"Dataset statistics saved to: {output_path}")

    def collate_fn(self, samples):
        """Batch processing function for data loader"""
        if not samples:
            return {}
        
        # remove text-related keys and two-stage keys
        text_keys = {"description"}
        two_stage_keys = {"large_partial_input", "num_large_objects"}
        string_keys = {"data_source"}  # Add data_source to excluded keys
        scene_graph_keys = {"scene_graph"}  # Add scene_graph to excluded keys
        # Keep object_size_masks in the processing (it should be processed as a 1D array)
        key_set = set(samples[0].keys()) - {"length"} - text_keys - two_stage_keys - string_keys - scene_graph_keys
        
        # use fixed sequence length for padding
        # Note: samples already include start and end tokens, so no additional adjustment needed
        if self.encoding_type == "diffusion":
            max_length = self.config.get("sample_num_points", self._max_length)
        elif self.encoding_type == "autoregressive":
            max_length = max(sample["length"] for sample in samples)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # keys that need padding (2D arrays)
        padding_keys = set(k for k in key_set if len(samples[0][k].shape) == 2)
        sample_params = {}
        
        # keys that don't need padding
        sample_params.update({
            k: np.stack([sample[k] for sample in samples], axis=0)
            for k in (key_set - padding_keys)
        })
        
        # keys that need padding
        sample_params.update({
            k: np.stack([
                np.vstack([
                    sample[k],
                    np.zeros((max_length - len(sample[k]), sample[k].shape[1]))
                ]) for sample in samples
            ], axis=0)
            for k in padding_keys
        })
        
        sample_params["lengths"] = np.array([
            sample["length"] for sample in samples
        ])
        
        # Handle two-stage training data
        if self.two_stage_small_from_large and all("large_partial_input" in sample for sample in samples):
            # Find max number of large objects across the batch
            max_large_objects = max(sample["num_large_objects"] for sample in samples)
            
            if max_large_objects > 0:
                # Get the feature dimension from the first non-empty large_partial_input
                feature_dim = None
                for sample in samples:
                    if sample["num_large_objects"] > 0:
                        feature_dim = sample["large_partial_input"].shape[1]
                        break
                
                if feature_dim is not None:
                    # Pad large_partial_input to same size across batch
                    large_partial_inputs = []
                    for sample in samples:
                        large_input = sample["large_partial_input"]
                        if len(large_input) < max_large_objects:
                            # Pad with zeros
                            padding = np.zeros((max_large_objects - len(large_input), feature_dim))
                            large_input = np.vstack([large_input, padding])
                        large_partial_inputs.append(large_input)
                    
                    sample_params["large_partial_input"] = np.stack(large_partial_inputs, axis=0)
                    sample_params["num_large_objects"] = np.array([sample["num_large_objects"] for sample in samples])
                else:
                    # All samples have empty large objects
                    sample_params["large_partial_input"] = np.zeros((len(samples), 0, 1))
                    sample_params["num_large_objects"] = np.zeros(len(samples))
            else:
                # No large objects in any sample
                sample_params["large_partial_input"] = np.zeros((len(samples), 0, 1))
                sample_params["num_large_objects"] = np.zeros(len(samples))
        
        # process description - check if all samples have description key
        if all("description" in sample for sample in samples):
            sample_params["description"] = [sample["description"] for sample in samples]
        
        # convert to PyTorch tensors
        torch_sample = {
            k: torch.from_numpy(sample_params[k]).float()
            for k in sample_params if k != "description"
        }
        
        # add extra dimension for translations
        torch_sample.update({
            k: torch_sample[k][:, None]
            for k in torch_sample.keys()
            if "_tr" in k
        })
        
        # add description (if exists) - description remains as string list
        if "description" in sample_params:
            # Use Any type to avoid type checking issues
            torch_sample["description"] = sample_params["description"]  # type: ignore
        
        # add data_source (if exists) - data_source remains as string list
        if all("data_source" in sample for sample in samples):
            torch_sample["data_source"] = [sample["data_source"] for sample in samples]
        
        # add scene_graph (if exists) - scene_graph remains as list of dicts
        if all("scene_graph" in sample for sample in samples):
            torch_sample["scene_graph"] = [sample["scene_graph"] for sample in samples]
        
        return torch_sample

    # property accessors, keep compatibility with existing code
    @property
    def object_types(self):
        return self._object_types

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def class_frequencies(self):
        return self._class_frequencies

    @property
    def class_order(self):
        return self._class_order

    @property
    def count_furniture(self):
        return self._count_furniture

    @property
    def max_length(self):
        return self._max_length

    @property
    def n_object_types(self):
        return len(self._object_types)

    @property
    def n_classes(self):
        return len(self._class_labels)

    @property
    def bounds(self):
        return {
            "translations": self.centroids,
            "sizes": self.sizes,
            "angles": self.angles,
        }

    @property
    def centroids(self):
        return self._centroids

    @property
    def sizes(self):
        return self._sizes

    @property
    def angles(self):
        return self._angles

    @property
    def feature_size(self):
        """Feature size, compatible with existing encoding system"""
        if self.encoding_type == "diffusion":
            return 8 + self.n_classes  # 3(translation) + 3(size) + 2(cos/sin angle) + n_classes
        elif self.encoding_type == "autoregressive":
            return 7 + self.n_classes  # 3(translation) + 3(size) + 1(angle) + n_classes
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    @property
    def bbox_dims(self):
        """Bounding box dimensions"""
        return 8  # 3(translation) + 3(size) + 2(cos/sin angle)

    def __str__(self):
        return f"M3DLayoutDataset contains {len(self.scenes)} scenes with {self.n_object_types} discrete types"
