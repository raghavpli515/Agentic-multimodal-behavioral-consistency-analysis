import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel
print("FUSION_MODEL_RELOADED_12345")

# =========================================
# VIDEO MODEL
# =========================================
class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v2(pretrained=True)

        self.features = backbone.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1280, 128)

        self.classifier = nn.Linear(128, 5)

    def forward(self, x, return_embedding=False):

        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)

        feat = self.features(x)
        feat = self.pool(feat)
        feat = feat.view(feat.size(0), -1)

        feat = self.fc(feat)

        feat = feat.view(B, T, -1)

        feat = feat.mean(dim=1)

        if return_embedding:
            return feat

        return self.classifier(feat)
    
# =========================================
# AUDIO MODEL
# =========================================
class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, 128)

        self.classifier = nn.Linear(128, 5)

    def forward(self, x, return_embedding=False):

        print("\n===== AUDIO INPUT =====")
        print("Shape:", x.shape)
        print("Device:", x.device)
        print("Dtype:", x.dtype)

        print("Min:", x.min().item())
        print("Max:", x.max().item())

        print("NaN:", torch.isnan(x).any().item())
        print("Inf:", torch.isinf(x).any().item())

        print("Contiguous:", x.is_contiguous())

        x = x.contiguous()
        print("LSTM INPUT SHAPE:", x.shape)
        print("LSTM DEVICE:", x.device)
        print("LSTM DTYPE:", x.dtype)

        try:
            out, _ = self.lstm(x)
        except Exception as e:

            print("SHAPE:", x.shape)

            print("MIN:", x.min())

            print("MAX:", x.max())

            print("NAN:", torch.isnan(x).any())

            print("INF:", torch.isinf(x).any())

            raise e

        out = out[:, -1, :]

        feat = self.fc(out)

        if return_embedding:
            return feat

        return self.classifier(feat)
    

# =========================================
# TEXT MODEL
# =========================================
class TextModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        for param in self.bert.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(768, 128)

        self.classifier = nn.Linear(128, 5)

    def forward(self, input_ids, attention_mask,
                return_embedding=False):

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = out.last_hidden_state[:, 0, :]

        feat = self.fc(cls)

        if return_embedding:
            return feat

        return self.classifier(feat)

# =========================================
# FUSION MODEL
# =========================================
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio_model = AudioModel()
        self.text_model = TextModel()
        self.video_model = VideoModel()

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(
        self,
        audio,
        input_ids,
        attention_mask,
        video,
        return_all=False
    ):

        audio_feat = self.audio_model(
            audio,
            return_embedding=True
        )

        text_feat = self.text_model(
            input_ids,
            attention_mask,
            return_embedding=True
        )

        video_feat = self.video_model(
            video,
            return_embedding=True
        )

        fused = torch.cat([
            audio_feat,
            text_feat,
            video_feat
        ], dim=1)

        fusion_logits = self.classifier(fused)

        audio_logits = self.audio_model.classifier(audio_feat)
        text_logits = self.text_model.classifier(text_feat)
        video_logits = self.video_model.classifier(video_feat)

        if return_all:
            return {
                "fusion_logits": fusion_logits,
                "audio_logits": audio_logits,
                "text_logits": text_logits,
                "video_logits": video_logits
            }

        return fusion_logits
