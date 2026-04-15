import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # Remove classifier, keep convolutional feature extractor
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        num_features = backbone.classifier.in_features

        self.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def extract_features(self, x):
        """Return convolutional feature maps before global pooling."""
        return self.feature_extractor(x)

    def forward_features(self, x):
        """Return normalized embedding and the last convolutional feature map."""
        features = self.extract_features(x)
        pooled = self.pool(features).view(features.size(0), -1)
        emb = self.fc(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb, features

    def forward_once(self, x):
        emb, _ = self.forward_features(x)
        return emb

    def forward(self, img1, img2):
        return self.forward_once(img1), self.forward_once(img2)

    def compute_sfam(self, feat1, feat2, normalize=True, eps=1e-8):
        """Compute Similar Feature Activation Map between two feature maps.

        Produces channel-wise cosine similarities at each spatial location,
        useful for visualising which regions drove a similarity prediction.
        """
        if feat1.shape != feat2.shape:
            raise ValueError("Feature maps must have the same shape for SFAM computation")

        similarity = (feat1 * feat2).sum(dim=1, keepdim=True)
        norm1 = torch.norm(feat1, dim=1, keepdim=True)
        norm2 = torch.norm(feat2, dim=1, keepdim=True)
        sfam = similarity / (norm1 * norm2 + eps)

        if normalize:
            sfam = (sfam + 1.0) / 2.0
        return sfam

    def forward_with_sfam(self, img1, img2, output_size=None):
        """Forward two images and return embeddings + their SFAM."""
        emb1, feat1 = self.forward_features(img1)
        emb2, feat2 = self.forward_features(img2)
        sfam = self.compute_sfam(feat1, feat2)

        if output_size is not None:
            sfam = F.interpolate(sfam, size=output_size, mode="bilinear", align_corners=False)

        return emb1, emb2, sfam

    def get_embedding(self, x):
        return self.forward_once(x)
