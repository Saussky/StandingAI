import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import crop_and_preprocess_image_pil, crop_image_pil, preprocess_image_pil


## Data loading and preprocessing

base_dir = "./images"

# Define transformations including data augmentation
transform = transforms.Compose(
    [
        transforms.Lambda(
            lambda x: crop_and_preprocess_image_pil(
                x, crop_image_pil, preprocess_image_pil, target_size=(224, 224)
            )
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]
)

# Load dataset
dataset = datasets.ImageFolder(base_dir, transform=transform)

# Split dataset into training and validation
train_idx, val_idx = train_test_split(
    list(range(len(dataset))), test_size=0.2, random_state=42
)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)


## Model definition

# Load pre-trained MobileNetV2
mobilenet = models.mobilenet_v2(pretrained=True)
for param in mobilenet.parameters():
    param.requires_grad = False  # Freeze parameters

# Modify classifier
mobilenet.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(mobilenet.last_channel, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 1),
    nn.Sigmoid(),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet.to(device)


## Training

criterion = nn.BCELoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    num_epochs=20,
):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()

            outputs = model(inputs)
            preds = outputs > 0.5
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(
            f"Epoch {epoch}/{num_epochs - 1} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
        )

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)

            outputs = model(inputs)
            preds = outputs > 0.5
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)

        print(
            f"Epoch {epoch}/{num_epochs - 1} Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )


train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    num_epochs=20,
)


# import torch
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# from utils import crop_and_preprocess_image_pil, crop_image_pil, preprocess_image_pil


# def visualize_preprocessed_images(
#     image_path, crop_function, preprocess_function, target_size=(224, 224)
# ):
#     """
#     Load an image, preprocess it using the specified functions, and visualize the preprocessed image.
#     Args:
#     - image_path: Path to the image file.
#     - crop_function: Function to crop the image.
#     - preprocess_function: Function to preprocess the image (after cropping).
#     - target_size: The target size for preprocessing.
#     """
#     # Load the image
#     image = Image.open(image_path).convert("RGB")

#     # Apply the preprocessing directly as done in the transform
#     preprocessed_img_tensor = crop_and_preprocess_image_pil(
#         image, crop_function, preprocess_function, target_size
#     )

#     # Convert the preprocessed tensor back to a PIL image for visualization
#     # This step correctly handles the tensor without assuming an incorrect shape
#     unloader = transforms.ToPILImage()
#     image_pil = unloader(
#         preprocessed_img_tensor.squeeze(0)
#     )  # Remove batch dimension if present

#     # Display the image
#     plt.imshow(image_pil)
#     plt.title("Preprocessed Image")
#     plt.axis("off")
#     plt.show()


# # Example usage with a specific image and preprocessing functions
# image_path = "./images/standing/standing2.jpg"  # Adjust the path to an actual image
# visualize_preprocessed_images(
#     image_path, crop_image_pil, preprocess_image_pil, target_size=(224, 224)
# )
