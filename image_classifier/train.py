#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from get_args import *
from get_data import *
from model import *

    
def main():
    # get input params
    in_arg = get_input_args_for_train()
    arch = in_arg.arch
    hidden_units = in_arg.hidden_units
    gpu = in_arg.gpu
    learning_rate = in_arg.learning_rate
    epochs = in_arg.epochs
    print_every = in_arg.print_every
    data_dir = in_arg.data_dir
    
    # cteate model object
    mymodel = myModel(arch, hidden_units, gpu, True)
    # load pretrained model
    model = mymodel.load_pretrained_model()
    # redefine the classifier
    model = mymodel.redefine_model(model)
    
    # get data set
    trainloader, validloader, testloader, class_to_idx = get_data(data_dir)
    
    # Train the model
    mymodel.train_model(model, trainloader, validloader, learning_rate, epochs, print_every)

    # Test the model
    mymodel.test_model(model, testloader)
    
    # Save the model
    mymodel.save_model(model, class_to_idx)
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()