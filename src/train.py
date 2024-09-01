import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from src.gs_utils.gs_dataset_utils import load_gs_dataset, GSDataset
from src.model.modeling_gst import GSTModel, GSTConfig
from src.gs_utils.training_utils import add_noise, loss_fn, cosine_beta_schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_diffusion_model(model, train_dataloader, eval_dataloader, training_args):
    optimizer = optim.Adam(model.parameters(), lr=training_args.lr)
    total_steps = len(train_dataloader) * training_args.epochs
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )
    
    for epoch in tqdm(range(training_args.epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            optimizer.zero_grad()        

            noisy_data, weight = add_noise(batch)

            noisy_data = noisy_data.to(device)
            outputs = model(noisy_data)

            # Calculate the loss (MSE between original and predicted clean data)
            loss = loss_fn(outputs, batch, weight)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{training_args.epochs}], Loss: {epoch_loss/len(train_dataloader):.4f}")

        # Evaluate the model
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                noisy_data, weight = add_noise(batch)
                outputs = model(noisy_data)
                loss = loss_fn(outputs, batch, weight)
                eval_loss += loss.item()
        
        print(f"Eval Loss: {eval_loss/len(eval_dataloader):.4f}")
    
    print("Training Complete")


class TrainingArguments:
    def __init__(self, **kwargs):
        self.epochs = kwargs.get("epochs", 10)
        self.lr = kwargs.get("lr", 1e-4)
        self.batch_size = kwargs.get("batch_size", 32)
        self.warmup_ratio = kwargs.get("warmup_ratio", 0.1)


def main():

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # Load the dataset
    train_dataset, test_dataset = load_gs_dataset()

    # Tokenize dataset
    train_dataloader = torch.utils.data.DataLoader(GSDataset(train_dataset), batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(GSDataset(test_dataset), batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    config = GSTConfig()
    model = GSTModel(config).to(device)

    # Print model parameter number
    print("Model parameters: ", sum(p.numel() for p in model.parameters()))

    # Train the model
    train_diffusion_model(model,
        train_dataloader,
        test_dataloader,
        TrainingArguments(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )
    )

    # Save the model
    torch.save(model.state_dict(), "gs_model.pth")

    print("Model saved")

if __name__ == "__main__":
    main()
