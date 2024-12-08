import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Generator
torch.autograd.set_detect_anomaly(True)
class Generator(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, num_residual_blocks=6):
        super(Generator, self).__init__()
        layers = [
            nn.Conv1d(input_dim, hidden_dim, kernel_size=15, padding=7),
            nn.InstanceNorm1d(hidden_dim),
            nn.ReLU(inplace=False)
        ]

        # Downsampling
        for _ in range(2):
            layers.append(
                nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2)
            )
            layers.append(nn.InstanceNorm1d(hidden_dim * 2))
            layers.append(nn.ReLU(inplace=False))
            hidden_dim *= 2

        # Residual blocks
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(hidden_dim))

        # Upsampling
        for _ in range(2):
            layers.append(
                nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=5, stride=2, padding=2, output_padding=1)
            )
            layers.append(nn.InstanceNorm1d(hidden_dim // 2))
            layers.append(nn.ReLU(inplace=False))
            hidden_dim //= 2

        layers.append(nn.Conv1d(hidden_dim, input_dim, kernel_size=15, padding=7))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm1d(dim),
            nn.ReLU(inplace=False),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm1d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim=80):
        super(Discriminator, self).__init__()
        layers = [
            nn.Conv1d(input_dim, 128, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2, inplace=False),
        ]

        num_filters = 128
        for _ in range(3):
            layers.append(
                nn.Conv1d(num_filters, num_filters * 2, kernel_size=15, stride=2, padding=7)
            )
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            num_filters *= 2

        layers.append(nn.Conv1d(num_filters, 1, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, min_length=None, max_length=None):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]
        
        # Load and filter spectrograms based on length
        self.spectrograms = []
        for file in self.files:
            try:
                spec = np.load(file)
                
                # Handle various possible shapes
                if spec.ndim == 4:
                    spec = spec.squeeze()
                
                # Ensure shape is (80, time_steps)
                if spec.ndim == 3:
                    if spec.shape[0] == 1:
                        spec = spec.squeeze(0)
                    elif spec.shape[2] == 80:
                        spec = spec.transpose(0, 2).squeeze(0)
                elif spec.ndim == 2:
                    if spec.shape[1] == 80:
                        spec = spec.T
                
                # Verify shape
                assert spec.shape[0] == 80, f"Unexpected spectrogram shape for {file}: {spec.shape}"
                
                # Apply length filtering if specified
                if min_length is not None and spec.shape[1] < min_length:
                    continue
                if max_length is not None and spec.shape[1] > max_length:
                    continue
                
                self.spectrograms.append(spec)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx]  # Return NumPy array directly

def pad_to_max_size(tensor1, tensor2):
    # Get the maximum size along the time dimension
    max_size = max(tensor1.shape[2], tensor2.shape[2])
    
    # Pad both tensors to the max size
    tensor1 = torch.nn.functional.pad(tensor1, (0, max_size - tensor1.shape[2]))
    tensor2 = torch.nn.functional.pad(tensor2, (0, max_size - tensor2.shape[2]))
    
    return tensor1, tensor2

# Utility function to pad or trim tensors to a consistent length
def pad_or_trim_tensor(tensor, target_length):
    # Handle 4D tensor from some spectrogram formats
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # Removing the extra dimension

    # If tensor is already a PyTorch tensor, convert to NumPy if needed
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()

    # Ensure tensor is of shape (80, time_steps)
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif tensor.shape[2] == 80:
            tensor = tensor.transpose(0, 2).squeeze(0)
    elif tensor.ndim == 2:
        # If shape is (time_steps, 80)
        if tensor.shape[1] == 80:
            tensor = tensor.T

    # Ensure shape is (80, time_steps)
    assert tensor.shape[0] == 80, f"Unexpected tensor shape: {tensor.shape}"

    current_length = tensor.shape[1]

    if current_length == target_length:
        return torch.from_numpy(tensor)

    if current_length < target_length:
        # Pad with zeros
        pad_size = target_length - current_length
        padded = np.pad(tensor, ((0, 0), (0, pad_size)), mode='constant')
        return torch.from_numpy(padded)
    else:
        # Trim 
        return torch.from_numpy(tensor[:, :target_length])  

# Loss Functions
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
def load_models(model_dir, G_A2B, G_B2A, D_A, D_B):
    # Find the latest epoch
    saved_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    epochs = [int(f.split('_epoch_')[1].split('.pth')[0]) for f in saved_files]
    if len(epochs) == 0:
        print("No saved models found. Starting training from scratch.")
        return 0  # Start from epoch 0

    latest_epoch = max(epochs)
    # Load the weights of the latest models
    G_A2B.load_state_dict(torch.load(os.path.join(model_dir, f'G_A2B_epoch_{latest_epoch}.pth')))
    G_B2A.load_state_dict(torch.load(os.path.join(model_dir, f'G_B2A_epoch_{latest_epoch}.pth')))
    D_A.load_state_dict(torch.load(os.path.join(model_dir, f'D_A_epoch_{latest_epoch}.pth')))
    D_B.load_state_dict(torch.load(os.path.join(model_dir, f'D_B_epoch_{latest_epoch}.pth')))

    #print(f"Loaded models from epoch {latest_epoch}")
    return latest_epoch

def train_cycle_gan_resume(train_A_dir, train_B_dir, output_dir, model_dir, epochs=400, batch_size=2, lr=0.0002, patience=8):
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets and determine common spectrogram length
    dataset_A = SpectrogramDataset(train_A_dir)
    dataset_B = SpectrogramDataset(train_B_dir)
    
    print(f"Dataset A: {len(dataset_A)} spectrograms")
    print(f"Dataset B: {len(dataset_B)} spectrograms")
    
    def get_common_length(dataset):
        lengths = [spec.shape[1] for spec in dataset.spectrograms]
        return min(lengths)

    common_length = min(get_common_length(dataset_A), get_common_length(dataset_B))
    print(f"Common length: {common_length}")
    
    dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True, 
                               collate_fn=lambda x: torch.stack([pad_or_trim_tensor(spec, common_length) for spec in x]))
    dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True, 
                               collate_fn=lambda x: torch.stack([pad_or_trim_tensor(spec, common_length) for spec in x]))

    # Initialize models
    G_A2B = Generator()
    G_B2A = Generator()
    D_A = Discriminator()
    D_B = Discriminator()

    # Load saved models if available
    start_epoch = load_models(model_dir, G_A2B, G_B2A, D_A, D_B)

    # Optimizers
    opt_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
    opt_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # Learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, mode='min', patience=patience, factor=0.5, verbose=True)
    scheduler_D_A = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_D_A, mode='min', patience=patience, factor=0.5, verbose=True)
    scheduler_D_B = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_D_B, mode='min', patience=patience, factor=0.5, verbose=True)

    # Keep track of the last two saved model files
    last_saved_files = []
    best_loss = float('inf')  # Initialize best loss to infinity
    best_model_files = None  # To track model files for the lowest loss
    
    log_file = os.path.join(output_dir, "training_log.txt")

    # Training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss_G = 0
        total_loss_D_A = 0
        total_loss_D_B = 0
        
        progress_bar = tqdm(zip(dataloader_A, dataloader_B), total=min(len(dataloader_A), len(dataloader_B)))
        for real_A, real_B in progress_bar:
            # Generate fake data
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            cycle_A = G_B2A(fake_B)
            cycle_B = G_A2B(fake_A)

            cycle_A, real_A = pad_to_max_size(cycle_A, real_A)
            cycle_B, real_B = pad_to_max_size(cycle_B, real_B)

            # Loss calculations
            loss_cycle = l1_loss(cycle_A, real_A) + l1_loss(cycle_B, real_B)
            loss_identity = l1_loss(G_A2B(real_B), real_B) + l1_loss(G_B2A(real_A), real_A)
            fake_B_output = D_B(fake_B).detach()
            loss_G_A2B = mse_loss(fake_B_output, torch.ones_like(fake_B_output))
            fake_A_output = D_A(fake_A).detach()
            loss_G_B2A = mse_loss(fake_A_output, torch.ones_like(fake_A_output))
            loss_G = loss_G_A2B + loss_G_B2A + 10 * loss_cycle + 5 * loss_identity

            # Train discriminators
            opt_D_A.zero_grad()
            loss_D_A = (mse_loss(D_A(real_A), torch.ones_like(D_A(real_A))) +
                        mse_loss(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))) * 0.5
            loss_D_A.backward()
            torch.nn.utils.clip_grad_norm_(D_A.parameters(), max_norm=1.0)  # Gradient clipping
            opt_D_A.step()

            opt_D_B.zero_grad()
            loss_D_B = (mse_loss(D_B(real_B), torch.ones_like(D_B(real_B))) +
                        mse_loss(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))) * 0.5
            loss_D_B.backward()
            torch.nn.utils.clip_grad_norm_(D_B.parameters(), max_norm=1.0)  # Gradient clipping
            opt_D_B.step()

            # Train generators
            opt_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(list(G_A2B.parameters()) + list(G_B2A.parameters()), max_norm=1.0)  # Gradient clipping
            opt_G.step()


            total_loss_G = total_loss_G + loss_G.item()
            total_loss_D_A = total_loss_D_A + loss_D_A.item()
            total_loss_D_B += total_loss_D_B + loss_D_B.item()

            progress_bar.set_description(f"Epoch {epoch + 1}/{start_epoch + epochs}")
            progress_bar.set_postfix({
                'Loss G': loss_G.item(), 
                'Loss D_A': loss_D_A.item(), 
                'Loss D_B': loss_D_B.item()
            })

        avg_loss_G = total_loss_G / len(progress_bar)
        avg_loss_D_A = total_loss_D_A / len(progress_bar)
        avg_loss_D_B = total_loss_D_B / len(progress_bar)

        print(f"Epoch {epoch + 1}/{start_epoch + epochs}")
        print(f"Avg Loss G: {avg_loss_G:.4f}\tAvg Loss D_A: {avg_loss_D_A:.4f}\tAvg Loss D_B: {avg_loss_D_B:.4f}")

        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}/{start_epoch + epochs}\n")
            f.write(f"Avg Loss G: {avg_loss_G:.4f}\tAvg Loss D_A: {avg_loss_D_A:.4f}\tAvg Loss D_B: {avg_loss_D_B:.4f}\n")
        
        # Step learning rate schedulers
        scheduler_G.step(avg_loss_G)
        scheduler_D_A.step(avg_loss_D_A)
        scheduler_D_B.step(avg_loss_D_B)

        # Save models
        model_files = [
            (G_A2B, f'G_A2B_epoch_{epoch + 1}.pth'),
            (G_B2A, f'G_B2A_epoch_{epoch + 1}.pth'),
            (D_A, f'D_A_epoch_{epoch + 1}.pth'),
            (D_B, f'D_B_epoch_{epoch + 1}.pth'),
        ]

        current_model_paths = []
        for model, filename in model_files:
            path = os.path.join(output_dir, filename)
            torch.save(model.state_dict(), path)
            current_model_paths.append(path)
            last_saved_files.append(path)
            
        # Keep track of best loss and corresponding models
        if avg_loss_G < best_loss:
            best_loss = avg_loss_G
            best_model_files = current_model_paths.copy()
            print(f"New best model found at epoch {epoch + 1} with loss {best_loss:.4f}")
            with open(log_file, "a") as f:
                f.write(f"New best model found at epoch {epoch + 1} with loss {best_loss:.4f}\n")
        # Keep only the last two saved files, excluding best model files
        while len(last_saved_files) > 8:
            file_to_remove = last_saved_files.pop(0)
            if best_model_files and file_to_remove not in best_model_files:
                os.remove(file_to_remove)

                
                
        # Early stopping check
        #if avg_loss_G < best_loss_G:
         #   best_loss_G = avg_loss_G
          #  epochs_no_improve = 0
        #else:
        #    epochs_no_improve += 1

        #if epochs_no_improve >= patience:
         #   print(f"Early stopping triggered after {epoch + 1} epochs. Best Loss G: {best_loss_G:.4f}")
          #  break


# Paths
train_A_dir = "sp/kaggle/working/specs/spec/arijit/"
train_B_dir = "sp/kaggle/working/specs/spec/kishore/"
model_dir = "trained_models/"
output_dir = "saved_models/"

# Continue training
train_cycle_gan_resume(train_A_dir, train_B_dir, output_dir,model_dir)