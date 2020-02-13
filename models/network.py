"""defines the network architecture
given the limited number of images in the train set a pretrained network
will be used for the feature extraction...

the base model in considareation is the "mobilenet_v2"
https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
"""
import torch
import torchvision.models as models

class Network(torch.nn.Module):
    """classificator of cars
    """
    def __init__(self, opt):
        self.opt = opt
        self.model = build_network()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.params,
                                          lr=self.opt.lr,
                                          betas=(self.opt.beta1, self.opt.beta2))

    def build_network(self):
        """build our custom network over the mobilenet_v2
        """
        # load pretrained mobilenet_v2
        model = models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # the last module of the mobilenet_v2 is called classifier
            # (0): Dropout(p=0.2, inplace=False)
            # (1): Linear(in_features=1280, out_features=1000, bias=True)
        in_features = model.classifier[1].in_features
        # we add our custom fully connected layer according to the num of classes
        model.classifier[1] = torch.nn.Linear(in_features, self.opt.num_classes)
        return model

    def forward(self, x):
        return self.model(x)

    def test(self, x):
        """same as forward but with no backpropagation
        """
        with torch.no_grad():
            return self.model(x)

    def save(self, epoch):
        file_name = f'checkpoint_epoch_{epoch}_{self.opt.lr}'
        save_path = os.path.join(self.opt.checkpoint_path, file_name)
        torch.save(self.model.cpu().state_dict(), save_path)
