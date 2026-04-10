import torch
import torch.nn as nn
import torchvision.models as models

class VideoModel(nn.Module):
    def __init__(self, num_classes=5, hidden_dim=256):
        #The super() function is used to call the __init__ method of the parent class (nn.Module) 
        #to ensure that the VideoModel class is properly initialized as a PyTorch module. 
        # This allows us to use all the functionalities provided by nn.Module, such as parameter management and model saving/loading.
        super(VideoModel, self).__init__()    

        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])   #(*list(resnet.children())[:-1]) is used to create a new sequential module that contains all layers of the ResNet-50 model except the final fully connected layer. 
                                                                                #This allows us to use ResNet-50 as a feature extractor, where we take the output from the last convolutional layer (which has 2048 features) and feed it into our own LSTM and classifier for video classification tasks.

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.embedding_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_embedding=False):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)

        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(features)
        final_output = lstm_out[:, -1, :]

        if return_embedding:
            return final_output

        return self.classifier(final_output)