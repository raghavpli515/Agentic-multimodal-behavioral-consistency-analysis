import torch
import torch.nn as nn
from transformers import BertModel

class TextModel(nn.Module):
    def __init__(self, num_classes=5):
        super(TextModel, self).__init__()  

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Freeze BERT initially
        for param in self.bert.parameters():
            param.requires_grad = False

        self.embedding_dim = 768 

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 256), 
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, return_embedding=False):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask  #
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token

        if return_embedding:
            return cls_embedding

        return self.classifier(cls_embedding)