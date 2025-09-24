import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

class SqueezeNetBackbone(nn.Module):
    """
    SqueezeNet backbone for DETR.
    This module extracts feature maps from a SqueezeNet model, formats them for DETR,
    and handles the necessary mask propagation.
    """
    def __init__(self, pretrained: bool = True):
        """
        Initializes the SqueezeNet backbone.

        Args:
            pretrained (bool): If True, loads weights pretrained on ImageNet.
        """
        super().__init__()
        # Load the SqueezeNet 1.1 model from torchvision
        squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None)

        # We only need the 'features' part of SqueezeNet, which is the convolutional base.
        # The classifier and final pooling layers are discarded.
        self.features = squeezenet.features

        # The number of output channels from the last layer of the 'features' module.
        # For SqueezeNet 1.1, this is 512. DETR needs this to configure its transformer.
        self.num_channels = 512

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        """
        Forward pass through the backbone.

        Args:
            pixel_values (torch.Tensor): The input images (batch_size, num_channels, height, width).
            pixel_mask (torch.Tensor): A boolean mask for padding (batch_size, height, width).

        Returns:
            dict: A dictionary containing the output feature map and its corresponding mask,
                  formatted as DETR expects.
        """
        # Pass the tensors through the convolutional layers of SqueezeNet
        out_features = self.features(pixel_values)

        # The mask needs to be downsampled to the same spatial dimensions as the output
        # feature map. The total stride of SqueezeNet 1.1 is 32.
        mask = pixel_mask
        if mask is not None:
            # Use functional interpolation to resize the mask.
            # The mask is 2D, but interpolate needs a channel dimension, so we unsqueeze and squeeze.
            mask = F.interpolate(mask[None].float(), size=out_features.shape[-2:]).to(torch.bool)[0]

        # DETR expects the output in a specific dictionary format.
        # For a single feature map output, we use the key '0'.
        out = {'0': {'tensors': out_features, 'mask': mask}}

        return out

def main():
    """
    Example usage of the SqueezeNetBackbone with a Hugging Face DETR model.
    """
    print("Initializing custom SqueezeNet backbone...")
    # Initialize our custom backbone. Set pretrained=False to avoid downloading weights
    # during testing if you don't have them cached.
    backbone = SqueezeNetBackbone(pretrained=True)
    print(f"Backbone initialized. Output channels: {backbone.num_channels}")

    print("\nConfiguring DETR model to use the custom backbone...")
    # Create a custom DETR configuration.
    # The key steps are:
    # 1. Set `backbone=None` and `use_timm_backbone=False` to tell the model we are
    #    providing our own backbone module.
    # 2. Set `num_channels` to match the output channels of our backbone (512).
    config = DetrConfig(
        backbone=None,
        use_timm_backbone=False,
        num_channels=backbone.num_channels,
        use_pretrained_backbone=False,
        # You can adjust other DETR parameters here if needed
        num_queries=100,
        num_labels=91, # COCO class count + 1 for "no object"
    )
    

    print("Instantiating DetrForObjectDetection model...")
    # Instantiate the DETR model, passing both the custom config and the backbone instance.
    model = DetrForObjectDetection(config=config)
    # Manually set the backbone to our custom SqueezeNet module.
    model.backbone = backbone
    model.eval() # Set model to evaluation mode
    print("Model created successfully.")

    print("\nPreparing a dummy input image...")
    # The image processor handles resizing and normalization.
    image_processor = DetrImageProcessor()
    # Create a dummy image with random data.
    # Dimensions are (batch, channels, height, width).
    dummy_image = torch.randint(0, 256, (1, 3, 480, 640)).float()
    inputs = image_processor(images=dummy_image, return_tensors="pt")

    print("Running a forward pass through the model...")
    with torch.no_grad():
        outputs = model(**inputs)

    print("Forward pass successful!")
    print("\n--- Output Shapes ---")
    print(f"Logits (predictions): {outputs.logits.shape}")
    print(f"Boxes (predictions):  {outputs.pred_boxes.shape}")
    print("---------------------\n")


if __name__ == "__main__":
    main()



