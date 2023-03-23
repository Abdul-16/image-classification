import torch
import numpy as np
from torch import nn, optim
from data_utils import DataUtils
from torchvision import models
from collections import OrderedDict

class ModelUtils:
    
    def __init__(self, data_dir='flowers', arch='vgg16', hidden_units=[4096,512], learning_rate=0.01, gpu=True):
        
        self.data_utils = DataUtils(data_dir)
        self.trainloader, self.validloader, self.testloader, self.train_data = self.data_utils.load_data()
        self.hidden_units = hidden_units
        self.arch = arch
        self.model = getattr(models, arch)(pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        if arch == 'vgg16':
            num_features = 25088
            
        elif arch == 'densenet121':
            num_features = 1024
        
        for param in self.model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_features, hidden_units[0])),
                                                ('relu1', nn.ReLU()),
                                                ('dropout1', nn.Dropout(p=0.2)),
                                                ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                                                ('relu2', nn.ReLU()),
                                                ('dropout2', nn.Dropout(p=0.2)),
                                                ('fc3', nn.Linear(hidden_units[1], 128)),
                                                ('output', nn.LogSoftmax(dim=1))
                                                ]))
            
        self.model.classifier = classifier
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.classifier.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train_model(self, epochs=10):
        steps = 0
        running_loss = 0
        print_every = 50
        best_loss = np.Inf
        
        for epoch in range(epochs):
            for inputs, labels in self.trainloader:
                steps += 1

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                logps = self.model(inputs)
                loss = self.criterion(logps, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    self.model.eval()

                    with torch.no_grad():
                        for inputs, labels in self.validloader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)

                            logps = self.model(inputs)
                            batch_loss = self.criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Training loss: {running_loss/print_every:.3f} | "
                          f"Validation loss: {valid_loss/len(self.validloader):.3f} | "
                          f"Validation accuracy: {accuracy/len(self.validloader):.3f}")

                    if valid_loss/len(self.validloader) < best_loss:
                        best_loss = valid_loss/len(self.validloader)
                        torch.save(self.model.state_dict(), 'best_model.pt')

                    running_loss = 0
                    self.model.train()
        
        print('Training complete.')
        return self.model
        
    def save_checkpoint(self, trained_model, save_dir, epochs=10):
        trained_model.class_to_idx = self.train_data.class_to_idx

        if self.arch == 'vgg16':
            num_features = 25088
            checkpoint = {'arch': 'vgg16',
             'input_size': num_features,
             'output_size': 128,
             'hidden_layers': self.hidden_units,
             'state_dict': trained_model.state_dict(),
             'class_to_idx': self.trainloader.dataset.class_to_idx,
             'optimizer_state': self.optimizer.state_dict(),
             'epochs': epochs}
            
        elif self.arch == 'densenet121':
            num_features = 1024
            checkpoint = {'arch': 'densenet121',
             'input_size': num_features,
             'output_size': 128,
             'hidden_layers': self.hidden_units,
             'state_dict': self.model.state_dict(),
             'class_to_idx': self.trainloader.dataset.class_to_idx,
             'optimizer_state': self.optimizer.state_dict(),
             'epochs': epochs}

        torch.save(checkpoint, save_dir)
        print(f'Checkpoint saved at {save_dir}')
        
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        
        if checkpoint['arch'] == 'vgg16':
            model = models.vgg16(pretrained=True)
            num_features = model.classifier[0].in_features
        
        elif checkpoint['arch'] == 'densenet121':
            model = models.densenet121(pretrained=True)
            num_features = model.classifier.in_features
        
        for param in model.parameters():
            param.requires_grad = False
        model.class_to_idx = checkpoint['class_to_idx']
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num_features, checkpoint['hidden_layers'][0])),
                                                ('relu1', nn.ReLU()),
                                                ('dropout1', nn.Dropout(p=0.2)),
                                                ('fc2', nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1])),
                                                ('relu2', nn.ReLU()),
                                                ('dropout2', nn.Dropout(p=0.2)),
                                                ('fc3', nn.Linear(checkpoint['hidden_layers'][1], 128)),
                                                ('output', nn.LogSoftmax(dim=1))
                                                ]))

        
        model.classifier = classifier

        state_dict = checkpoint['state_dict']

        model.load_state_dict(state_dict)
        
        return model