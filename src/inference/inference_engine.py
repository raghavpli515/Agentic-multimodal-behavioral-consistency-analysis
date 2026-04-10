# Inference Engine for Multimodal Sentiment Analysis, which loads the trained models and processes new video inputs to predict sentiment probabilities for each modality and their fusion.
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from src.models.video_model import VideoModel
from src.models.audio_model import AudioModel
from src.models.text_model import TextModel
from src.models.fusion_model import FusionModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os

class InferenceEngine:

    def __init__(self):

        # Load models
        self.video_model = VideoModel().to(device)
        self.audio_model = AudioModel().to("cpu")  # Load audio model on CPU to save GPU memory, since audio processing is less intensive and can be done on CPU without significant performance loss.
        self.text_model = TextModel().to(device)

        self.fusion_model = FusionModel(         # The FusionModel is initialized with the embedding dimensions of the video and audio models. This allows the fusion model to properly combine the features extracted from both modalities. The embedding_dim attribute of each model is used to specify the size of the feature vectors that will be input into the fusion model.
            video_dim=self.video_model.embedding_dim,
            audio_dim=self.audio_model.embedding_dim
        ).to(device)

        # Load weights
        # Get project root
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

        MODEL_DIR = os.path.join(ROOT_DIR, "models_saved")

        self.video_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "video_model.pt"), map_location=device))
        self.audio_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "audio_model.pt"), map_location=device))
        self.text_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "text_model.pt"), map_location=device))
        self.fusion_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "fusion_model.pt"), map_location=device))

        self.video_model.eval() #.eval() sets the model to evaluation mode, which is important for layers like dropout and batch normalization to behave correctly during inference.
        self.audio_model.eval()
        self.text_model.eval()
        self.fusion_model.eval()

    def predict(self, video_frames, audio_mfcc=None, text_ids=None, text_mask=None):

        with torch.no_grad():    # torch.no_grad() is a context manager that disables gradient calculation, which is useful during inference to save memory and computational resources since we don't need to compute gradients for backpropagation when making predictions.

            video_frames = video_frames.to(device)

            # ---- Video ----
            video_logits = self.video_model(video_frames)
            video_probs = F.softmax(video_logits, dim=1).cpu().numpy()[0]

            # ---- Handle Missing Modalities ----
            if audio_mfcc is None:
                audio_probs = np.zeros_like(video_probs)
                audio_embed = torch.zeros((1, self.audio_model.embedding_dim)).to(device)
            else:
                audio_mfcc = audio_mfcc.to("cpu")  # Process audio on CPU to save GPU memory, since audio processing is less intensive and can be done on CPU without significant performance loss.
                audio_logits = self.audio_model(audio_mfcc)
                audio_probs = F.softmax(audio_logits, dim=1).cpu().numpy()[0]
                audio_embed = self.audio_model(audio_mfcc, return_embedding=True)
                audio_embed = audio_embed.to(device)  # Move audio embedding to GPU for fusion

            if text_ids is None or text_mask is None:
                text_probs = np.zeros_like(video_probs)
            else:
                text_ids = text_ids.to(device)
                text_mask = text_mask.to(device)
                text_logits = self.text_model(text_ids, text_mask)
                text_probs = F.softmax(text_logits, dim=1).cpu().numpy()[0]

            # ---- Fusion ----
            video_embed = self.video_model(video_frames, return_embedding=True)

            fusion_logits = self.fusion_model(video_embed, audio_embed)
            fusion_probs = F.softmax(fusion_logits, dim=1).cpu().numpy()[0]

        return {
            "video_probs": video_probs.tolist(),
            "audio_probs": audio_probs.tolist(),
            "text_probs": text_probs.tolist(),
            "fusion_probs": fusion_probs.tolist()
        }
    
    def preprocess_frames(self, frames):
        transform = transforms.Compose([      #transforms.Compose is a function from the torchvision.transforms module that allows you to chain together multiple image transformations. In this case, it is used to define a sequence of transformations that will be applied to each video frame before feeding them into the model.
            transforms.ToPILImage(),          #transforms.ToPILImage() converts a NumPy array (which is the format of the video frames read by OpenCV) into a PIL Image object, which is a common format for image processing in PyTorch.
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        processed = [transform(frame) for frame in frames]

        video_tensor = torch.stack(processed[:8])  # (T, C, H, W)   #torch.stack is used to combine a list of tensors (in this case, the processed video frames) into a single tensor. The resulting video_tensor will have a shape of (T, C, H, W), where T is the number of frames, C is the number of channels (e.g., 3 for RGB), and H and W are the height and width of the frames after resizing.
        video_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)   #unsqueeze(0) adds a new dimension at the beginning of the tensor, which is often used to represent the batch size. In this case, it converts the video_tensor from shape (T, C, H, W) to (1, T, C, H, W), indicating that we have a batch of 1 video sequence to process through the model.

        return video_tensor

    

    