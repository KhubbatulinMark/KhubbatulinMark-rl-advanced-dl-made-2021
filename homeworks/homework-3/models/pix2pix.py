import os
import torch
from .base import BaseModel
from .networks import GANLoss, define_G, define_D

class Pix2Pix(BaseModel):

    def __init__(self, model_name, params, device='cuda'):
        
        BaseModel.__init__(self, model_name, params)
        self.device = device
        
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.phase == 'train':
            self.model_names = ['G', 'D']
        else:  
            self.model_names = ['G']
        self.netG = define_G(3, 3, 64, 'unet_256', 'batch', True, 'normal', 0.02, [0])
        if self.phase == 'train':
            self.netD = define_D(6, 64, 'basic', 3, 'batch', 'normal', 0.02, [0])
            self.criterionGAN = GANLoss('vanilla').to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=params['lr'], betas=(params['beta'], 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=params['lr'], betas=(params['beta'], 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['B'].to(self.device)
        self.real_B = input['A'].to(self.device)
        self.image_paths = input['B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) 
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 100
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()    