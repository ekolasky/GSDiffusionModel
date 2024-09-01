import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from gs_utils.gs_dataset_utils import load_gs_dataset

def train_diffusion_model(model, dataloader, epochs=10, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # Add Gaussian noise to the batch data
            noisy_data = add_noise(batch)

            # Forward pass through the model
            outputs = model(noisy_data)

            # Calculate the loss (MSE between original and predicted clean data)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
    
    print("Training Complete")
    return model

def add_noise(batch, noise_level=0.1):
    # Add Gaussian noise to the GS ellipsoids
    noise = torch.randn_like(batch) * noise_level
    return batch + noise

def main():

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Load the dataset
    train_dataset, test_dataset = load_gs_dataset()

    # Tokenize dataset
    tokenized_train_dataset = tokenize_dataset(train_dataset)
    tokenized_test_dataset = tokenize_dataset(test_dataset)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = GSModel()

    # Train the model
    model = train_diffusion_model(model, train_dataloader, epochs=10, lr=1e-4)

    # Save the model
    torch.save(model.state_dict(), "gs_model.pth")

    print("Model saved")