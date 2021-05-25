import torch
import torch.nn as nn
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Discriminator, Generator


def main():
    # Hyper parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 64
    lr = 3e-4
    
    img_dim = 28 * 28
    batch_size = 32
    num_epochs = 50
    
    # Model
    discriminator = Discriminator(img_dim).to(device)
    generator = Generator(z_dim, img_dim).to(device)
    
    # Pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Data
    trainset = datasets.MNIST('dataset', transform=transform, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr)
    opt_gen = torch.optim.Adam(generator.parameters(), lr=lr)
    
    bce_loss = nn.BCELoss()
    writer = SummaryWriter(f"runs/GAN_MNIST/fake")
    
    # Logging and Tensorboard 
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    epoch_iterator = tqdm(range(num_epochs))
    
    for epoch in epoch_iterator:
        for batch_idx, (real, _) in enumerate(trainloader):
            real = real.view(-1, img_dim).to(device)
            batch_size = real.shape[0]
            
            # Train descriminator: minimize the classification loss
            noise = torch.randn((batch_size, z_dim)).to(device)
            fake = generator(noise)
            
            disc_real = discriminator(real).view(-1)
            disc_fake = discriminator(fake).view(-1)
            
            loss_real = bce_loss(disc_real, torch.ones_like(disc_real))
            loss_fake = bce_loss(disc_fake, torch.ones_like(disc_fake))
            
            disc_loss = (loss_real + loss_fake) / 2
            
            discriminator.zero_grad()
            disc_loss.backward(retain_graph=True)
            opt_disc.step()
            
            # Train generator: maximize the classification loss for fake images
            output = discriminator(fake).view(-1)
            gen_loss = bce_loss(output, torch.ones_like(output))
            generator.zero_grad()
            gen_loss.backward()
            opt_gen.step()
            
            # Tensorboard
            if batch_idx == 0:
                with torch.no_grad():
                    # Reshape to (B, C, H, W)
                    fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                    real = real.reshape(-1, 1, 28, 28)
                    
                    img_grid_real = torchvision.utils.make_grid(
                        real, 
                        normalize=True
                    )
                    
                    img_grid_fake = torchvision.utils.make_grid(
                        fake, 
                        normalize=True
                    )
                    
                    writer.add_scalar(
                        "Generator Loss", gen_loss, global_step=epoch
                    )
                    writer.add_scalar(
                        "Descriminator Loss", disc_loss, global_step=epoch
                    )
                    
                    writer.add_image(
                        "MNIST Fake Image", img_grid_fake, global_step=epoch
                    )
                    
                    writer.add_image(
                        "MNIST Real Image", img_grid_real, global_step=epoch
                    )
                    
        epoch_iterator.set_description(
            f"Loss D: {disc_loss:.4f}, Loss G: {gen_loss:.4f}"
        )
        
    

if __name__ == "__main__":
    main()
