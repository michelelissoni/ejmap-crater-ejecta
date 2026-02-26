"""
Filename: smp_custom_models.py
Author: Michele Lissoni
Date: 2026-02-13
"""

"""

Create segmentation neural network from smp (Segmentation Models Pytorch) library.

"""


import torch.nn as nn

from terratorch.models.model import ModelOutput

import segmentation_models_pytorch as smp

# The available models and their smp objects
SMP_dictionary = { "Unet" : smp.Unet,
                   "Unet++" : smp.UnetPlusPlus,
                   "FPN" : smp.FPN,
                   "PSPNet" : smp.PSPNet,
                   "DeepLabV3" : smp.DeepLabV3,
                   "DeepLabV3+" : smp.DeepLabV3Plus,
                   "Linknet" : smp.Linknet,
                   "MAnet" : smp.MAnet,
                   "UPerNet" : smp.UPerNet,
                   "Segformer" : smp.Segformer
                 }

# This module makes it possible to embed a data vector into a tensor outputted from a CNN

class InjectVector(nn.Module):
    def __init__(self, vector_dim, tensor_channels_list):
        """
        Args:
            tensor_channels_list (list of int): Number of channels for each tensor.
            vector_dim (int): Dimension of the conditioning vector.
        """
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(vector_dim, out_channels)
            for out_channels in tensor_channels_list
        ])

        self.activation = nn.ReLU()

    def forward(self, vector, features):
        """
        Args:
            tensor_list (list of tensors): Each tensor of shape (B, C_i, H, W)
            vector (tensor): (B, D)

        Returns:
            List of tensors: Each of shape (B, C_i, H, W) after adding the embedded vector.
        """
        out = []
        for tensor, linear in zip(features, self.linears):
            B, C, H, W = tensor.shape
            embedded = self.activation(linear(vector))   # (B, C)
            embedded = embedded.view(B, C, 1, 1)         # (B, C, 1, 1)
            embedded = embedded.expand(-1, -1, H, W)     # (B, C, H, W)
            out.append(tensor + embedded)
            
        return out

# An smp segmentation models where additional data (midinput) is inserted between the encoder and the decoder

class SMP_midinput(nn.Module):

    def __init__(self, model_name = "Unet", in_channels=4, midinput_dim=4, accept_midinput=True, **kwargs):
    
        """
        Initialization.
        
        Keywords:
        - model_name: one of the keys in `SMP_dictionary`.
        - in_channels (int): the number of channels in the image.
        - midinput_dim (int): the length of the midinput vector.
        - accept_midinput (bool): set to False if the model should not accept a midinput.
        - **kwargs: keywords fed to the SMP object.
        
        """
    
        super().__init__()
        
        model_Class = SMP_dictionary[model_name]

        model_orig = model_Class(in_channels=in_channels, classes=1, activation=None, **kwargs)
        
        self.encoder = model_orig.encoder
        encoder_out_channels = self.encoder.out_channels

        self.decoder = model_orig.decoder

        self.head = model_orig.segmentation_head

        self.accept_midinput = accept_midinput

        if(accept_midinput):
            self.injector = InjectVector(midinput_dim, encoder_out_channels[1:])

    def forward(self, x):
        
        image = x["image"]

        encoder_output = self.encoder(image)

        if(self.accept_midinput):
            midinput = x["midinput"]
            injector_output = self.injector(midinput, encoder_output[1:])
            injector_output = [encoder_output[0], *injector_output]
            decoder_output = self.decoder(*injector_output)

        else:
            decoder_output = self.decoder(*encoder_output)

        output = self.head(decoder_output)[:,0,:,:]

        return ModelOutput(output)
