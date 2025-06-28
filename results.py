# results of Generated images
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from scipy.linalg import sqrtm
from PIL import Image
import os


# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# List of image paths and corresponding prompts
image_prompt_pairs = [
    
    ( "15.png", "a boy and a girl playing football in the park on the middle of the image "),
    ("17.png", "a boy and a girl playing football in the park"),
    ("23.png", "a boy on the left and a girl on the right with a white background"),
    ("25.png", "a boy on the left and a girl on the right holding hands with a black background"),
    ("26.png", "a boy on the left side and a girl on the right side, both holding hands with  green background"),
    ("27.png", "one airplane boy and girl play soccer ball"),
    ("28.png", "a boy on the left side with a green background"), 
    ("30.png", "a boy in the park on the right and a cloud on the left upper corner in the sky"),
    
    

]

# List to store results
clip_scores = []

# Process each image and prompt
for image_path, prompt in image_prompt_pairs:
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        continue
    
    # Load and process image
    image = Image.open(image_path)
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    
    # Forward pass through the CLIP model
    outputs = model(**inputs)
    
    # Extract text and image embeddings
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    
    # Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity
    similarity = (image_embeds @ text_embeds.T).squeeze().item()
    clip_scores.append(similarity)
    
    print(f"CLIPScore for '{prompt}' and {image_path}: {similarity}")

# Calculate average CLIPScore
if clip_scores:
    avg_clip_score = sum(clip_scores) / len(clip_scores)
    print(f"\nAverage CLIPScore: {avg_clip_score}")
else:
    print("No valid images processed.")

# Load InceptionV3 model (pretrained on ImageNet)
inception = models.inception_v3(pretrained=True, transform_input=False)
inception.fc = torch.nn.Identity()  # Remove final classification layer
inception.eval()  # Set to evaluation mode

# Define preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Inception expects 299x299 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    """Extract features from the InceptionV3 model."""
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return None
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        features = inception(image)  # Get feature vector
    
    return features.cpu().numpy().squeeze()

def calculate_fid_single(feature1, feature2):
    """Compute a simplified FID-like distance for two images."""
    if feature1 is None or feature2 is None:
        return None
    
    mu1, mu2 = feature1, feature2  # Use feature vectors directly
    fid = np.sum((mu1 - mu2) ** 2)  # Squared Euclidean distance
    return fid

# List of image pairs (generated, original)
image_pairs = [
    
    ("happy girl.png", "Dataset-images/original-smilegirl.png"),
]

# Store FID scores
fid_scores = []

# Process each image pair
for gen_path, orig_path in image_pairs:
    features_gen = extract_features(gen_path)
    features_orig = extract_features(orig_path)
    
    fid_score = calculate_fid_single(features_gen, features_orig)
    if fid_score is not None:
        fid_scores.append(fid_score)
        print(f"FID Score for {gen_path} vs {orig_path}: {fid_score:.4f}")

# Calculate average FID score
if fid_scores:
    avg_fid_score = sum(fid_scores) / len(fid_scores)
    print(f"\nAverage FID Score: {avg_fid_score:.4f}")
else:
    print("No valid image pairs processed.")

