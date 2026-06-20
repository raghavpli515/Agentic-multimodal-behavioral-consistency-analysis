import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import os
torch.backends.cudnn.enabled = False
from src.models.fusion_model import FusionModel
from src.reasoning.trust_layer import TrustScorer

# =========================================
# DEVICE
# =========================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================
# INFERENCE ENGINE
# =========================================
class InferenceEngine:

    def __init__(self):

        ROOT_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )

        MODEL_DIR = os.path.join(ROOT_DIR, "checkpoints")

        MODEL_PATH = os.path.join(
            MODEL_DIR,
            "video_personal_best.pt"
        )
        print(torch.cuda.get_device_name(0))
        print(torch.version.cuda)
        print(torch.backends.cudnn.version())
        # =====================================
        # MODEL
        # =====================================
        self.model = FusionModel().to(device)

        state_dict = torch.load(
            MODEL_PATH,
            map_location=device
        )

        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

        # =====================================
        # TRUST SCORER
        # =====================================
        self.trust_scorer = TrustScorer()

        print("Multimodal model loaded")

    # =========================================
    # PREDICT
    # =========================================
    def predict(
        self,
        video_frames,
        audio_mfcc,
        text_ids,
        text_mask
    ):

        with torch.no_grad():

            video_frames = video_frames.to(device)
            if audio_mfcc is None:
                audio_mfcc = torch.zeros(
                    (1, 100, 40)
                ).float().to(device)
            audio_mfcc = audio_mfcc.to(device)
            if text_ids is None:
                text_ids = torch.zeros(
                    (1, 16),
                    dtype=torch.long
                ).to(device)

                text_mask = torch.zeros(
                    (1, 16),
                    dtype=torch.long
                ).to(device)

            if text_ids is not None:
                text_ids = text_ids.to(device)
            if text_mask is not None:
                text_mask = text_mask.to(device)

            outputs = self.model(
                audio_mfcc,
                text_ids,
                text_mask,
                video_frames,
                return_all=True
            )

            fusion_probs = F.softmax(
                outputs["fusion_logits"],
                dim=1
            ).cpu().numpy()[0]

            audio_probs = F.softmax(
                outputs["audio_logits"],
                dim=1
            ).cpu().numpy()[0]

            text_probs = F.softmax(
                outputs["text_logits"],
                dim=1
            ).cpu().numpy()[0]

            video_probs = F.softmax(
                outputs["video_logits"],
                dim=1
            ).cpu().numpy()[0]


            # =====================================
            # TRUST SCORING
            # =====================================
            trust_result = self.trust_scorer.compute(
                fusion_probs,
                audio_probs,
                text_probs,
                video_probs
            )

            prediction = int(np.argmax(fusion_probs))

            torch.cuda.empty_cache()

        return {
            "prediction": prediction,
            "fusion_probs": fusion_probs.tolist(),
            "audio_probs": audio_probs.tolist(),
            "text_probs": text_probs.tolist(),
            "video_probs": video_probs.tolist(),
            "trust_analysis": trust_result,
            "modal_predictions": {
                "audio": int(np.argmax(audio_probs)),
                "text": int(np.argmax(text_probs)),
                "video": int(np.argmax(video_probs))
            }
        }



    # =========================================
    # FRAME PREPROCESS
    # =========================================
    def preprocess_frames(self, frames):

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])

        processed = [
            transform(frame)
            for frame in frames
            if frame is not None
        ]

        if len(processed) == 0:
            raise ValueError("No valid frames extracted")

        processed = processed[:6]

        video_tensor = torch.stack(processed)
        video_tensor = video_tensor.unsqueeze(0)

        return video_tensor