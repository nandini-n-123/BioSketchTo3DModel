import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def main():
    print("Setting up data pipelines...")
    
    # 1. TRAIN Transforms (Heavy augmentation to teach the AI to handle messiness)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30), 
        transforms.RandomPerspective(p=0.5), 
        transforms.ElasticTransform(alpha=50.0), # The trick for messy hand-drawings
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. TEST Transforms (NO augmentation! Just resize and clean formatting)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    try:
        # Load Training Data
        train_dataset = datasets.ImageFolder(root='dataset/train', transform=train_transforms)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Load Testing Data (The unseen images you just moved!)
        test_dataset = datasets.ImageFolder(root='dataset/test', transform=test_transforms)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        print(f"Training on {len(train_dataset)} images. Testing on {len(test_dataset)} images.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Make sure your dataset folder has both 'train' and 'test' subfolders!")
        return

    print("Initializing EfficientNet-B0...")
    # Load the pre-trained model
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    # Modify the final layer to output exactly 2 classes (Brain vs Hibiscus)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2) 

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 3
    print(f"\n--- Starting Training & Testing for {epochs} Epochs ---")
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train() # Tell PyTorch we are training
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # --- TESTING PHASE ---
        model.eval() # Tell PyTorch to lock the weights (no learning, just guessing)
        correct = 0
        total = 0
        
        with torch.no_grad(): # Disable gradient math to save memory and speed up testing
            for images, labels in test_dataloader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1) # Get the index of the highest prediction
                total += labels.size(0)
                correct += (predicted == labels).sum().item() # Count how many we got right
        
        # Calculate Percentage Accuracy based on the Test folder
        accuracy = 100 * correct / total
        
        print(f">>> Epoch {epoch+1} | Train Loss: {running_loss/len(train_dataloader):.4f} | Validation Accuracy: {accuracy:.2f}%")

    # Save the Final Model 
    torch.save(model.state_dict(), 'biosketch_model.pt')
    print("\n🎉 Training Complete! Model saved successfully as 'biosketch_model.pt'!")

if __name__ == '__main__':
    main()