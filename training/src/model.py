import argparse
import os

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import copy


class BaseballCNN(nn.Module):
    def __init__(self,filters=32,dropout=0.1,nodes=32,kernel_size=3,dimensions=(216,192)):
        super().__init__()
        self.kernel_size_1=kernel_size
        self.dimensions=dimensions
        
        self.linear_mult= int((int((int((dimensions[0]-(kernel_size-1))/2) -2)/2) -2)/2) * int((int((int((dimensions[1]-(kernel_size-1))/2) -2)/2) -2)/2) 
        print(self.linear_mult)
        self.conv1 = nn.Conv2d(1, filters, kernel_size)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Conv2d(filters, filters, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(p=dropout)
        
        self.conv3 = nn.Conv2d(filters, filters, 3)
        self.pool3 = nn.MaxPool2d(2)
        
        self.linear_image = nn.Linear(filters*self.linear_mult,nodes-4)
        self.linear_meta = nn.Linear(20,4)
        self.drop3 = nn.Dropout(dropout)
        
        self.linear2 = nn.Linear(nodes,nodes) 
        self.drop4 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(nodes,1)
        
    def forward(self, x_image,x_meta):
        
        x_image = F.relu(self.conv1(x_image))
        x_image = self.pool1(x_image)
        x_image = self.drop1(x_image)
        # print(x_image.shape)
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        x_image = self.drop2(x_image)
        x_image = F.relu(self.conv3(x_image))
        # print(x_image.shape)
        x_image = self.pool3(x_image)
        # print(x_image.shape)
        x_image = torch.flatten(x_image,1)
        # print(x_image.shape)
        x_image = self.linear_image(x_image)
        # print(x_image.shape)
        x_meta = self.linear_meta(x_meta)
        # print(x_image.shape,x_meta.shape)
        x = torch.cat((x_image,x_meta),1)
        x= torch.sigmoid(self.linear3(self.drop4(self.linear2(x))))
        return x
    
    
    
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--filters', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--kernel-size', type=int, default=3)
    
    parser.add_argument('--use-cuda', type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    
    param_dict = copy.copy(vars(args))
    filters = param_dict["filters"]

    dropout = param_dict["dropout"]
    nodes = param_dict["nodes"]
    num_epochs = param_dict["epochs"]
    # learning_rate = 0.0005 
    learning_rate = args.learning_rate
    
    kernel_size = args.kernel_size
    
    batch_size = args.batch_size
    # device = torch.device('mps')
    # model = BaseballCNN().to(device)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda: ',torch.cuda.is_available())



    
    images = np.load(os.path.join(args.train, 'image_file.npy'))
    meta = np.load(os.path.join(args.train, 'meta_file.npy'))
    df = pd.read_csv(os.path.join(args.train, 'processed_pitch_table.csv'))
    
    print(images.shape)

    train_dataset = TensorDataset(torch.Tensor(images[:9000]/255),
                            torch.Tensor(meta[:9000]),
                            torch.Tensor(df.label.values[:9000]))


    test_dataset = TensorDataset(torch.Tensor(images[9000:]/255),
                            torch.Tensor(meta[9000:]),
                            torch.Tensor(df.label.values[9000:]))


    all_dataset = TensorDataset(torch.Tensor(images/255),
                            torch.Tensor(meta),
                            torch.Tensor(df.label.values))


    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)



    # Loss and optimizer
    criterion = nn.BCELoss()
    model = BaseballCNN(filters,dropout,nodes,kernel_size).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    train_losses=[]
    test_losses=[]
    test_acc=[]
    train_acc=[]
    # Train the model
    patience = 5
    lowest_loss = 1
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        print(epoch)
        epoch_train_losses=0
        epoch_test_losses=0
        epoch_train_acc=0
        epoch_test_acc=0
        model.train()
        for i, (image_batch,meta_batch, y_batch) in enumerate(train_loader):  

            optimizer.zero_grad()
            image_batch = image_batch.to(device)
            meta_batch  = meta_batch.to(device)
            y_batch     = y_batch.to(device)

            # Forward pass and loss calculation
            outputs = model(image_batch,meta_batch)
            loss = criterion(outputs.squeeze(1), y_batch)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Validation Loop 

        with torch.no_grad(): 
            model.eval() 
            for i, (image_batch,meta_batch, y_batch) in enumerate(test_loader): 
                image_batch = image_batch.to(device)
                meta_batch  = meta_batch.to(device)
                y_batch     = y_batch.to(device)
                outputs = model(image_batch,meta_batch)
                loss = criterion(outputs.squeeze(1), y_batch)
                epoch_test_losses+=outputs.shape[0]*loss.item()
                epoch_test_acc += (((outputs[:,0].cpu().detach().numpy()>0.5)*1==y_batch.cpu().detach().numpy())*1).sum()

            for i, (image_batch,meta_batch, y_batch) in enumerate(train_loader): 
                image_batch = image_batch.to(device)
                meta_batch  = meta_batch.to(device)
                y_batch     = y_batch.to(device)
                outputs = model(image_batch,meta_batch)
                loss = criterion(outputs.squeeze(1), y_batch)
                epoch_train_losses+=outputs.shape[0]*loss.item()
                epoch_train_acc += (((outputs[:,0].cpu().detach().numpy()>0.5)*1==y_batch.cpu().detach().numpy())*1).sum()


        # print(f'Validation Loss: {running_val_loss/len(test_loader):.4f}')

    
        test_losses += [epoch_test_losses/len(test_dataset)]
        test_acc += [epoch_test_acc/len(test_dataset)]
        train_losses += [epoch_train_losses/len(train_dataset)]
        train_acc += [epoch_train_acc/len(train_dataset)]

        print(f'train_loss={train_losses[-1]};')
        print(f'train_acc={train_acc[-1]};')
        print(f'test_loss={test_losses[-1]};')
        print(f'test_acc={test_acc[-1]};')
        
        if (test_losses[-1] < lowest_loss) or (epoch < 20):
            lowest_loss = test_losses[-1]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > patience:
            print('Early stopping')
            break

        
       
        # Additional information
        EPOCH = epoch
        PATH = f"model_large_{EPOCH}.pt"

        torch.save({
                    'epoch': EPOCH,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_losses[-1],
                    }, PATH)
        
            # ... train `model`, then save it to `model_dir`
    print('test loss : ',test_losses)
    print('test acc  : ',test_acc)
    print('train loss: ',train_losses)
    print('train acc : ',train_acc)
    print(f"Test set: Average loss: {np.min(test_losses)};")
    best_epoch = np.argmin(test_losses)
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)