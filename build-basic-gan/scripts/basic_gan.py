# Libraries
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

# Generator

## Generator Block
def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

  class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
      super(Generator, self).__init__()
      # Build Neural Network
      self.gen =  nn.Sequential(
        get_generator_block(z_dim, hidden_dim),
        get_generator_block(hidden_dim, hidden_dim*2),
        get_generator_block(hidden_dim*2, hidden_dim*4),
        nn.Linear(hidden_dim*4, im_dim),
        nn.Sigmoid()
      )
      
      def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)
      
      def get_gen(self):
        return self.gen


      # Noise
def get_noise(n_samples, z_dim, device='cpu'):
  '''
  Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
  creates a tensor of that shape filled with random numbers from the normal distribution.
  Parameters:
    n_samples: the number of samples to generate, a scalar
    z_dim: the dimension of the noise vector, a scalar
    device: the device type
  '''
  return torch.randn(size = (n_samples, z_dim), device=device)

# Discriminator

## Discriminator Block
def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
      nn.Linear(input_dim, output_dim),
      nn.LeakyRELU(negative_slope=0.2)
    )
  
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=784, hidden_dim=128):
      super(Discriminator, self)__init__()
      self.disc = nn.Sequential(
        get_discriminator_block(im_dim, hidden_dim*4),
        get_discriminator_block(hidden_dim*4, hidden_dim*2),
        get_discriminator_block(hidden_dim*2, hidden_dim),
        nn.Linear(hidden_dim, 1)
      )
      
    def forward(self, image):
      '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
      return self.disc(image)
    
    def get_disc(self):
      return self.disc

# Training

## parameter setup

criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

device = 'cuda'

## Initialize Models
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

## Losses
### discriminator loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    noise = get_noise(num_images, z_dim, device=device)
    gen_imgs = gen(noise)
    pred_fake = disc(gen_imgs.detach()) # NOTE: SHould be detached
    pred_real = disc(real)
    loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake)) # Discriminator wants to correctly identify fake images, hence 0
    loss_real = criterion(pred_real, torch.ones_like(pred_real)) # Discriminator wants to correctly identify real, hence 1
    # discriminator loss is the average of 'loss_fake' and 'loss_real'
    disc_loss = (loss_fake+loss_real)/2
    return disc_loss

### generator loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    noise = get_noise(num_images, z_dim, device=device)
    fake_images = gen(noise)
    fake_pred = disc(fake_images)
    gen_loss = criterion(fake_images, torch.ones_like(fake_pred)) # NOTE: Here we treat fake labels as 1
    return gen_loss
  
## training pipeline


cur_step=0
mean_generator_loss=0
mean_discriminator_loss=0
test_generator=True
gen_loss=False
error=False
for epoch in range(n_epochs):
  for real,_ in tqdm(dataloader):
    cur_batch_size = len(real)
    # Flatten the images
    real = real.view(cur_batch_size,-1).to(device)
    ## Update Discriminator
    disc_opt.zero_grad() # Zero out the gradients before backpropagation
    disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device) # Get discriminator loss
    disc_loss.backward(retain_graph=True) # Update gradients
    disc_opt.step() # Update optimizer
    
    
    # FOr testing generator weights
    if test_generator:
      old_generator_weights = gen.gen[0][0].weight.detach().clone()
    ## Update Generator
    gen_opt.zero_grad()
    gen_loss = get_gen_loss(gen, disc, criterion, num_images, z_dim, device)
    gen_loss.backward(retain_graph=True)
    gen_opt.step()
    
    ## Test to check whether generator weights get updated
    if test_generator:
      try:
        assert lr > 2e-7 or (gen.gen[0][0].weight.grad.abs().max() < 5e-4 and epoch == 0)
        assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
      except:
        error = True
        print("Runtime tests have failed!")
    # Observe Discriminator Loss after set display steps
    mean_discriminator_loss += disc_loss.item() / display_step
    # Observe Generator Loss
    mean_generator_loss += gen_loss.item() / display_step
    
    # Visualization
    if cur_step % display_step ==0 and cur_step > 0:
      print(f"Step {cur_step}: Generator Loss: {mean_generator_loss}, Discriminator Loss: {mean_discriminator_loss}")
      fake_noise = get_noise(cur_batch_size, z_dim, device)
      fake_images = gen(fake_noise)
      show_tensor_images(fake)
      show_tensor_images(real)
      # Reset the loss calculater per display step
      mean_generator_loss = 0
      mean_discriminator_loss = 0
      cur_step +=1
