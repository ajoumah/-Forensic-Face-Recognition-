import torch
import torch.nn as nn
import torchvision.models as models


class GenericEmbeddingModel(nn.Module):

    """
    Universal embedding wrapper for:
    - CNNs
    - ViTs
    - Torchvision models
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        normalize: bool = True
    ):

        super().__init__()

        self.backbone_name = backbone_name
        self.normalize = normalize

        self.model = self._load_backbone(
            backbone_name,
            pretrained
        )

    # =====================================
    # LOAD MODEL
    # =====================================
    def _load_backbone(
        self,
        backbone_name,
        pretrained
    ):

        # =================================
        # RESNET FAMILY
        # =================================
        if backbone_name == "resnet18":

            model = models.resnet18(
                pretrained=pretrained
            )

            model = nn.Sequential(
                *list(model.children())[:-1]
            )

        elif backbone_name == "resnet50":

            model = models.resnet50(
                pretrained=pretrained
            )

            model = nn.Sequential(
                *list(model.children())[:-1]
            )

        elif backbone_name == "resnet101":

            model = models.resnet101(
                pretrained=pretrained
            )

            model = nn.Sequential(
                *list(model.children())[:-1]
            )

        # =================================
        # VGG FAMILY
        # =================================
        elif backbone_name == "vgg16":

            base_model = models.vgg16(
                pretrained=pretrained
            )

            model = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )

        elif backbone_name == "vgg19":

            base_model = models.vgg19(
                pretrained=pretrained
            )

            model = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )

        # =================================
        # EFFICIENTNET
        # =================================
        elif backbone_name == "efficientnet_b0":

            base_model = (
                models.efficientnet_b0(
                    pretrained=pretrained
                )
            )

            model = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )

        elif backbone_name == "efficientnet_b4":

            base_model = (
                models.efficientnet_b4(
                    pretrained=pretrained
                )
            )

            model = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )

        # =================================
        # VISION TRANSFORMERS
        # =================================
        elif backbone_name == "vit_b_16":

            model = models.vit_b_16(
                pretrained=pretrained
            )

        elif backbone_name == "vit_l_16":

            model = models.vit_l_16(
                pretrained=pretrained
            )

        else:

            raise ValueError(
                f"Unsupported backbone: "
                f"{backbone_name}"
            )

        return model

    # =====================================
    # FORWARD
    # =====================================
    def forward(self, x):

        x = self.model(x)

        # CNN output
        if len(x.shape) == 4:

            x = x.view(
                x.size(0),
                -1
            )

        # Normalize embedding
        if self.normalize:

            x = nn.functional.normalize(
                x,
                p=2,
                dim=1
            )

        return x