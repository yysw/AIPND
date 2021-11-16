import argparse

def get_input_args_for_train():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
        
    parser.add_argument('data_dir', type=str, default='flowers',help='path to folder of images')

    parser.add_argument('--save_dir', type=str, default='./', help='directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg19_bn', choices=('vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19_bn','vgg19'),
                        help='the CNN model architecture')

    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=1024, help='hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='training epochs')
    parser.add_argument('--print_every', type=int, default=5, help='print info every n steps')
    parser.add_argument('--gpu', action='store_true', default=True, dest='gpu',help='Use GPU for training')
    

    return parser.parse_args()

def get_input_args_for_predict():
    parser = argparse.ArgumentParser()
        
    parser.add_argument('image_path', type=str, help='image path')
    parser.add_argument('checkpoint', type=str, help='the saved model file')
    parser.add_argument('--arch', type=str, default='vgg13', choices=('vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19_bn','vgg19'),
                        help='the CNN model architecture')
    parser.add_argument('--hidden_units', type=int, default=1024, help='hidden units')

    parser.add_argument('--top_k', type=int, default=3, help='top k most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, dest='gpu',help='Use GPU for predicting')
    

    return parser.parse_args()

in_arg = get_input_args_for_train()
print(in_arg)
