import argparse
from model_utils import ModelUtils

def main():
    parser = argparse.ArgumentParser(description='Train a neural network to classify flower images')
    
    parser.add_argument('data_dir', type=str, help='Path to folder containing training, validation, and test data')
    parser.add_argument('-s', '--save_dir', type=str, default='checkpoint.pth', help='Path to folder to save checkpoints')
    parser.add_argument('-a', '--arch', type=str, default='vgg16', help='Choose architecture: vgg16 or densenet121')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('-u', '--hidden_units', type=int, default=[4096,512], help='Number of hidden units in classifier (provide a list of two numbers only)')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU for training?')
    args = parser.parse_args()
    
      
    model_utils = ModelUtils(args.data_dir, args.arch, args.hidden_units, args.learning_rate, args.gpu)
    trained_model = model_utils.train_model(args.epochs)
    model_utils.save_checkpoint(trained_model, args.save_dir, args.epochs)
    
if __name__ == '__main__':
    main()