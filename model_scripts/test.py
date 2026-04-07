import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import os

def test_single_image(image_path, model_path='biosketch_model.pt'):
    # 1. Check if the file actually exists
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Could not find the file at '{image_path}'")
        print("Please check the path. TIP: Try moving the image into your 2Dto3D folder and just typing 'Screenshot 2026-04-05 193445.png'\n")
        return

    print(f"\nLoading image: {image_path}...")
    
    # 2. Setup formatting
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load the image
    try:
        image = Image.open(image_path).convert('RGB') 
    except Exception as e:
        print(f"❌ Error loading image! Error: {e}")
        return

    input_tensor = transform(image).unsqueeze(0)

    print("Loading AI Model 'biosketch_model.pt'...")
    # 4. Load the Architecture and Weights
    model = efficientnet_b0()
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2) 

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"❌ Error loading model! Make sure 'biosketch_model.pt' is in the same folder. Error: {e}")
        return
        
    model.eval() # Turn off training mode

    # 5. Make the Prediction
    classes = ['brain', 'hibiscus'] 

    with torch.no_grad(): 
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        prediction = classes[predicted_idx.item()]
        confidence_percent = confidence.item() * 100

    # 6. Print Results
    print("\n" + "="*40)
    print("🤖 AI PREDICTION RESULT 🤖")
    print("="*40)
    print(f"I am {confidence_percent:.2f}% sure this is a: {prediction.upper()}")
    print("="*40 + "\n")

if __name__ == '__main__':
    print("--- BioSketch-3D Tester ---")
    # The .strip() removes accidental quotes if you copy-paste a Windows path!
    user_file = input("📸 Please enter the name or path of your image file: ").strip('\"\'')
    test_single_image(user_file)