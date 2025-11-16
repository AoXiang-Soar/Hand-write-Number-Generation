import os.path
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

"""
Configuration
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
epoch = 5
learning_rate = 1e-3

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

"""
Core Components
"""
class Scheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        # Create a linearly increasing β sequence to control the intensity of noise addition
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas  # Coefficient for retaining the original signal
        # Calculate the cumulative product of α
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        # Directly calculate the noise level at any time step t
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars).to(device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars).to(device)

    def add_noise(self, x0, t, noise=None):
        """
        Add noise to x0 in time step t
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return xt, noise


# Adjust the level of features learned by changing the number of channels
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# Convert discrete time steps into continuous embedding vectors
# so that the neural network can understand temporal information
class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

    def timestep_embedding(self, t, dim):
        """
        Generate sin positional encoding
        """
        # Generate frequency factors for a geometric sequence,
        # ensuring that different dimensions correspond to different frequencies
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Take the outer product of the time step vector and the frequency vector to obtain phase information
        emb = t[:, None] * emb[None, :]
        # Apply sine and cosine functions to the phase separately and then combine them to form the complete encoding
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        # If the dimension is odd, add a zero vector to keep the dimensions consistent.
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

    def forward(self, t):
        t_emb = self.timestep_embedding(t, self.embedding_dim)
        return self.linear2(self.activation(self.linear1(t_emb)))


class ConditionalUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_channels)

    def forward(self, x, t_emb, c_emb):
        x = self.conv(x)
        t_emb = self.time_mlp(nn.SiLU()(t_emb))
        c_emb = self.cond_mlp(nn.SiLU()(c_emb))
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        c_emb = c_emb.unsqueeze(-1).unsqueeze(-1)
        return x + t_emb + c_emb  # Conditional Fusion


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=32, cond_emb_dim=10):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # Time Step and Conditional Embedding
        self.time_embed = TimestepEmbedding(time_emb_dim)
        self.cond_embed = nn.Linear(cond_emb_dim, time_emb_dim)

        # Encoder Path
        self.enc1 = ConditionalUNetBlock(in_channels, 64, time_emb_dim, time_emb_dim)
        self.enc2 = ConditionalUNetBlock(64, 128, time_emb_dim, time_emb_dim)
        self.enc3 = ConditionalUNetBlock(128, 256, time_emb_dim, time_emb_dim)

        # Middle Layer
        self.mid = ConditionalUNetBlock(256, 256, time_emb_dim, time_emb_dim)

        # Decoder Path
        self.dec3 = ConditionalUNetBlock(256 + 128, 128, time_emb_dim, time_emb_dim)
        self.dec2 = ConditionalUNetBlock(128 + 64, 64, time_emb_dim, time_emb_dim)
        self.dec1 = ConditionalUNetBlock(64, 32, time_emb_dim, time_emb_dim)  # No skip connections needed

        self.final_conv = nn.Conv2d(32, out_channels, 1)

        # Down/Up Sample
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, t, c):
        # Embedding Time Steps and Conditions
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c)

        # Encoder Path
        x1 = self.enc1(x, t_emb, c_emb)  # (B, 64, 28, 28)
        x2 = self.downsample(x1)  # (B, 64, 14, 14)
        x2 = self.enc2(x2, t_emb, c_emb)  # (B, 128, 14, 14)
        x3 = self.downsample(x2)  # (B, 128, 7, 7)
        x3 = self.enc3(x3, t_emb, c_emb)  # (B, 256, 7, 7)

        # Middle Layer
        x_mid = self.mid(x3, t_emb, c_emb)  # (B, 256, 7, 7)

        # Decoder Path (with skip connections)
        x4 = self.upsample(x_mid)  # (B, 256, 14, 14)
        x4 = torch.cat([x4, x2], dim=1)  # (B, 256+128=384, 14, 14)
        x4 = self.dec3(x4, t_emb, c_emb)  # (B, 128, 14, 14)

        x5 = self.upsample(x4)  # (B, 128, 28, 28)
        x5 = torch.cat([x5, x1], dim=1)  # (B, 128+64=192, 28, 28)
        x5 = self.dec2(x5, t_emb, c_emb)  # (B, 64, 28, 28)

        # Final Decoding Layer (no skip connections needed)
        x6 = self.dec1(x5, t_emb, c_emb)  # (B, 32, 28, 28)

        return self.final_conv(x6)  # (B, 1, 28, 28)


class DiffusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scheduler = Scheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        self.unet = UNet(in_channels=1, out_channels=1, time_emb_dim=32, cond_emb_dim=10)
        self.one_hot = DiffusionNet.tensor_2_one_hot

    @staticmethod
    def int_2_one_hot(x: int):
        tensor = torch.zeros(10, device=device)
        tensor[x] = 1
        return tensor

    @staticmethod
    def tensor_2_one_hot(x: torch.Tensor):
        return F.one_hot(x, num_classes=10).float().to(device)

    def forward(self, x, labels):
        c = self.one_hot(labels)
        t = torch.randint(0, self.scheduler.num_timesteps, (x.shape[0],), device=device)  # Generate time step
        noise = torch.randn_like(x, device=device)  # Generate random noise
        noisy_x, true_noise = self.scheduler.add_noise(x, t, noise)  # Forward diffusion
        t_normalized = t.float() / self.scheduler.num_timesteps
        predicted_noise = self.unet(noisy_x, t_normalized, c)  # Noise prediction
        loss = F.mse_loss(predicted_noise, true_noise)
        return loss

    def generate(self, num_samples, labels=None, return_all_steps=False):
        self.eval()
        with torch.no_grad():
            if labels is None:
                labels = torch.randint(0, 10, (num_samples,), device=device)
            c = self.one_hot(labels)
            x = torch.randn((num_samples, 1, 28, 28), device=device)  # Generate from random noise

            if return_all_steps:
                all_steps = [x.cpu()]

            for t in reversed(range(self.scheduler.num_timesteps)):
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.float32)
                t_normalized = t_batch / self.scheduler.num_timesteps
                predicted_noise = self.unet(x, t_normalized, c)

                alpha_t = self.scheduler.alphas[t]
                alpha_bar_t = self.scheduler.alpha_bars[t]
                beta_t = self.scheduler.betas[t]

                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha_t)) * (
                        x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
                ) + torch.sqrt(beta_t) * noise

                if return_all_steps:
                    all_steps.append(x.cpu())

            if return_all_steps:
                return torch.stack(all_steps)
            else:
                return x

"""
Train
"""
def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses = []

    for e in range(epoch):
        progress_bar = tqdm(train_loader, desc=f'Epoch {e + 1}/{epoch}')
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = model(images, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f'Epoch: {e + 1}, Average Loss: {avg_loss:.4f}')

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, e + 2), epoch_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.show()

        torch.save(model.state_dict(), 'diffusion.pkl')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), epoch_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    return epoch_losses


if __name__ == '__main__':
    model = DiffusionNet().to(device)

    if os.path.exists('diffusion.pkl'):
        mode = input("Load existing model? Training mode? (y/n): ")
        model.load_state_dict(torch.load('diffusion.pkl', map_location=device, weights_only=True))
        if mode.lower() == "y":
            train(model)
    else:
        print("No existing model found, starting training...")
        train(model)

    while True:
        label = input("Input a number to generate (0-9), or \"exit\" to exit: ")
        if label == "exit":
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            break
        elif label.isdigit() and 0 <= int(label) <= 9:
            generated = model.generate(num_samples=1, labels=torch.tensor([int(label)], device=device))
            plt.figure(figsize=(4, 4))
            img = generated.squeeze().cpu().numpy()
            plt.imshow(img, cmap='gray')
            plt.title(f'Generated Digit: {label}')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            print(f"Generated image for digit {label}")
        else:
            print("Please input a number between 0 and 9")