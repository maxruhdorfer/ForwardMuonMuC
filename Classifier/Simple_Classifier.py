import torch
import torch.nn as nn
from collections import OrderedDict
import os
import pandas as pd
import copy
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output

class SimpleClassifier(nn.Module):
    def __init__(self, imp, verbose = False):
        """Initialize the SimpleClassifier.

        Parameters:
            imp (list or str): Model architecture as a list of integers or a path to a saved model.
            verbose (bool): If True, print model structure.
        """
        super(SimpleClassifier, self).__init__()
        
        self.Instructions= 'This is the model for the classifier, to be later trained.\n\
Construct as c = SimpleClassifier(list), or c = SimpleClassifier(str).\n\
In the first case, the Architecture must be given as list of integers.\n\
In the second, the path to the saved model must be given as string. Starting from the notebook path.\n\
The .save(filename) method overwrites the filename file.\n\
The setscaling(features) function set the parameters of the scale function such that mean is zero and\n\
variance is one on the features dataset. The features input must be tensor([[features1], [features2], ...]).\n\
Since no weights are implemented here, scaling parameter must be set on the singnal only.\n\
Try verbose = True for more information' 
        
        if type(imp) == list:
            # Initialize architecture and scaling parameters
            self.Architecture = imp
            self.Me , self.St = torch.zeros(self.Architecture[0]), torch.ones(self.Architecture[0])
        elif type(imp) == str:
            # Load architecture and scaling parameters from a saved model
            self.Architecture = (torch.load(os.getcwd()+'/'+imp))[0]
            self.Me = (torch.load(os.getcwd()+'/'+imp))[1]
            self.St = (torch.load(os.getcwd()+'/'+imp))[2]
            self.Me = self.Me.cpu()
            self.St = self.St.cpu()
        else: 
            print('Input must be list or file name')
            return None
        
        # Add layers to the model
        for i in range(len(self.Architecture)-2):
            self.add_module(f"Lin_{i+1}", nn.Linear(self.Architecture[i], self.Architecture[i+1]))
            self.add_module(f"Sig_{i+1}", nn.Sigmoid())

        # Add the output layer
        self.add_module(f"Lin_{len(self.Architecture)-1}", nn.Linear(self.Architecture[-2], self.Architecture[-1]))
        if verbose: 
            print('for i in self.children(): print(i) command returns model structure, which is:\n')
            for i in self.children(): print(i)
            print('\n')
        
        if type(imp) == str:
            # Load model state if a model file was provided
            self.load_state_dict(
                (torch.load(os.getcwd()+'/'+imp))[3])
            if verbose: print('Loaded state_dict():\n',self.state_dict())
        
    def save(self,filename):
        """Save the model state and architecture.

        Parameters:
            filename (str): Name of the file to save the model.
        """
        torch.save([self.Architecture,self.Me,self.St,self.state_dict()], \
                   os.getcwd()+'/'+filename,_use_new_zipfile_serialization=False)
        
    def setscaling(self,FeaturesData):
        """Set scaling parameters based on the provided feature data.

        Parameters:
            FeaturesData (Tensor): Input feature data for scaling.
        """
        self.Me , self.St = torch.mean(FeaturesData,0), torch.std(FeaturesData,0) 
        if torch.get_default_dtype() == torch.float32: 
            self.Me , self.St = self.Me.float() , self.St.float()

    def scale(self,x):
        """Scale the input tensor using the mean and standard deviation.

        Parameters:
            x (Tensor): Input tensor to be scaled.

        Returns:
            Tensor: Scaled tensor.
        """
        return torch.mul(torch.add(x,-self.Me),1./self.St)
        
    def forward(self,x):
        """Forward pass through the network.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the network.
        """
        with torch.no_grad():
            f = self.scale(x)
            #print(f.device)
        for mod in self.children():
            f = mod(f)
        return f
    
class SimpleTrainer(nn.Module):
    def __init__(self, TrainingData, ValidationData,\
                 verbose = False,\
                 LearningRate = 1e-3, LossFunction = 'Quadratic', Optimiser = 'Adam', \
                 NumEpochs = 1000, \
                 PlotAfterEpoch = 10, \
                 SaveAtEpochs = None):
        """Initialize the SimpleTrainer.

        Parameters:
            TrainingData (list): Contains training features, weights, and labels.
            ValidationData (list): Contains validation features, weights, and labels.
            verbose (bool): If True, print additional information during training.
            LearningRate (float): Learning rate for the optimizer.
            LossFunction (str): Loss function to be used.
            Optimiser (str): Optimizer to be used.
            NumEpochs (int): Number of training epochs.
            PlotAfterEpoch (int): Frequency of plotting loss.
            SaveAtEpochs (list or None): Folder and epochs to save the model.
        """

        super(SimpleTrainer, self).__init__()

        # Instructions for using the trainer
        self.Instructions= 'This is the trainer, that contains all the information needed for training.\n\
Construct as t = SimpleTrainer(TrainingData,ValidationData).\n\
Data must have the form:\n\
    [ tensor([[features1], [features2], ...]) , \n\
      tensor([[weight1], [weight2], ... ]) , \n\
      tensor([[label1], [label2], ...  ])\n\n\
The most important training parameter is number of epochs, to be set as NumEpochs = int.\n\
It can be useful to save some models during training at some epochs.\n\
PlotAfterEpoch tells after how many epochs the plot should be updated.\n\
To set up: SaveAtEpochs = [ savefolder, [save epoch 1, save epoch 2, ...].\n\
If set up, SaveAtEpochs saves in the savefolder without deleting previous content.\n\
No save by default, but returns the trained classifier that can be saved.\n\n\
If p1 and p0 are the distributions of the 1 and 0 labeled data,\n\training will return a classifier:\n\
                r = Log[W1*p1/W0*p0]\n\n\
where W1 and W0 are the total weights of the training samples.\n\
The loss functions are normalized such that they give (W0+W1)/4 on impossible classification problem.\n\n\
Training works as: SimpleTrainer.train(SimpleClassiifer)\n\
In the plot, the validation loss is in red while training in blue.'
        
        # Assign training and validation data
        self.TrainingFeatures = TrainingData[0]
        self.TrainingWeights = TrainingData[1]
        self.TrainingLabels = TrainingData[2]
        
        self.ValidationFeatures = ValidationData[0]
        self.ValidationWeights = ValidationData[1]
        self.ValidationLabels = ValidationData[2]
        
        # Handle model saving configuration
        if SaveAtEpochs==None:
            self.SAtE = [ "none" , [-1] ]
        else: 
            self.SAtE = SaveAtEpochs
            try:
                os.mkdir(self.SAtE[0])
            except FileExistsError:
                pass
            
        self.NumberOfEpochs = NumEpochs
        
        # Define valid criteria and optimizers
        ValidCriteria = {'Quadratic': WeightedSELoss(), 'CrossEntropy': WeightedCELoss()}
        try:
            self.Criterion = ValidCriteria[LossFunction]
        except KeyError:
            print('The loss function specified is not valid. Allowed losses are %s.'
                 %str(list(ValidCriteria)))
        ValidOptimizers = {'Adam': torch.optim.Adam}
        try:
            self.Optimiser =  ValidOptimizers[Optimiser]
        except KeyError:
            print('The specified optimiser is not valid. Allowed optimisers are %s.'
                 %str(list(ValidOptimisers)))
            
        self.InitialLearningRate = LearningRate
        
        self.PlotEvery = PlotAfterEpoch
            
    def train(self, model, gpu = False, mini_batch_size = 100000):
        """Train the model using the provided training data.

        Parameters:
            model (nn.Module): The model to be trained.
            gpu (bool): If True, use GPU for training.
            mini_batch_size (int): Size of mini-batches for training.

        Returns:
            nn.Module: The trained model.
        """
        # Determine the data type of model parameters
        dtype = []
        for param in model.parameters():
            dtype.append(param.dtype)
        if len(dtype) == dtype.count(dtype[0]):
            typ = dtype[0]
        
        # Prepare training and validation data based on type
        if typ == torch.float32:
            TFeatures=self.TrainingFeatures.float()
            TLabels=self.TrainingLabels.float()
            TWeights=self.TrainingWeights.float()
            VFeatures=self.ValidationFeatures.float()
            VLabels=self.ValidationLabels.float()
            VWeights=self.ValidationWeights.float()
        elif typ == torch.float64:
            TFeatures=self.TrainingFeatures
            TLabels=self.TrainingLabels
            TWeights=self.TrainingWeights
            VFeatures=self.ValidationFeatures
            VLabels=self.ValidationLabels
            VWeights=self.ValidationWeights
        
        # Move model and data to GPU if specified
        if gpu:
            mod=copy.deepcopy(model)
            mod.cuda()
            #for i in mod.children(): 
             #   for j in i.parameters(): print(j.device)
            mod.Me=mod.Me.cuda()
            mod.St=mod.St.cuda()
            #print(mod.Me.device)
            #print(mod.St.device)

            
            TFeatures=TFeatures.cuda()
            TLabels=TLabels.cuda()
            TWeights=TWeights.cuda()
            VFeatures=VFeatures.cuda()
            VLabels=VLabels.cuda()
            VWeights=VWeights.cuda()
        else: mod=copy.deepcopy(model)
        
        # Initialize the optimizer
        Optimiser = self.Optimiser(mod.parameters(), self.InitialLearningRate)
        
        # Create plot data for loss
        Loss_Plot = LossPlot()
        
        for e in range(self.NumberOfEpochs):
            Optimiser.zero_grad()
            Tloss = 0.0

            # Training in mini-batches
            for b in range(0, TFeatures.size(0), mini_batch_size):
                TFeaturesb = TFeatures[b:b+mini_batch_size]
                TLabelsb = TLabels[b:b+mini_batch_size]
                TWeightsb = TWeights[b:b+mini_batch_size]
                Tclassifierb = mod.forward(TFeaturesb)
                Tlossb = self.Criterion(Tclassifierb, TLabelsb,TWeightsb)
                Tloss += Tlossb.tolist()
                Tlossb.backward()
            Optimiser.step()
            with torch.no_grad():
                for param in model.parameters():
                    param.clamp_(-5, 5)
            del Tlossb
            del Tclassifierb
            
            if (e+1) in self.SAtE[1]:
                # Save model at specified epochs
                mod.save(self.SAtE[0]+f"/AtEpoch_{e+1}")
            
            if e % self.PlotEvery == 0:
                # Calculate and plot validation loss
                with torch.no_grad():
                    Vloss = 0.0
                    for b in range(0, VFeatures.size(0), mini_batch_size):
                        VFeaturesb = VFeatures[b:b+mini_batch_size]
                        VLabelsb = VLabels[b:b+mini_batch_size]
                        VWeightsb = VWeights[b:b+mini_batch_size]
                        Vclassifierb = mod.forward(VFeaturesb)
                        Vlossb = self.Criterion(Vclassifierb, VLabelsb,VWeightsb)
                        Vloss += Vlossb.tolist()
                    Loss_Plot.UpdatePlots(e+1, Tloss,Vloss)
                    del Vlossb
                    del Vclassifierb
        # Move model back to CPU and return
        del Loss_Plot  
        mod.Me=mod.Me.cpu()
        mod.St=mod.St.cpu()
        return mod.cpu()
            
class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        """Base class for loss functions.

        Parameters:
            size_average (bool or None): Deprecated (see `reduction`).
            reduce (bool or None): Deprecated (see `reduction`).
            reduction (string, optional): Specifies the reduction to apply to the output.
        """
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
            
class WeightedSELoss(_Loss):
    __constants__ = ['reduction']
        
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        """Initialize weighted squared error loss.

        Parameters:
            size_average (bool or None): Deprecated (see `reduction`).
            reduce (bool or None): Deprecated (see `reduction`).
            reduction (string, optional): Specifies the reduction to apply to the output.
        """
        super(WeightedSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, classifier, labels, weights):
        """Calculate the weighted squared error loss.

        Parameters:
            classifier (Tensor): Output from the model.
            labels (Tensor): Ground truth labels.
            weights (Tensor): Weights for each sample.

        Returns:
            Tensor: Computed loss.
        """
        x = (1./(1.+ torch.exp(-classifier)))
        y = torch.mul(weights, (x-labels)**2)
        return torch.sum(y)
    
class WeightedCELoss(_Loss):
    __constants__ = ['reduction']
        
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        """Initialize weighted cross-entropy loss.

        Parameters:
            size_average (bool or None): Deprecated (see `reduction`).
            reduce (bool or None): Deprecated (see `reduction`).
            reduction (string, optional): Specifies the reduction to apply to the output.
        """
        super(WeightedCELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, classifier, labels, weights):
        """Calculate the weighted cross-entropy loss.

        Parameters:
            classifier (Tensor): Output from the model.
            labels (Tensor): Ground truth labels.
            weights (Tensor): Weights for each sample.

        Returns:
            Tensor: Computed loss.
        """
        x = (1./(1.+ torch.exp(-classifier)))
        y = torch.mul(weights, -torch.log(1.-labels + (2*labels-1.)*x))    
        return torch.sum(y)/np.log(2)/4.
    
class LossPlot():
    """Class for plotting training and validation loss."""

    def __init__(self):
        """Initialize the LossPlot class and set up the plot."""
        super(LossPlot, self).__init__()

        self.figLoss, self.axLoss  = plt.subplots()
        self.axLoss.set_yscale('log')

        # Display the plot dynamically in Jupyter
        self.hdisplay = display.display("", display_id = True)
        
        # Initialize empty plots for loss
        self.axLoss.plot([],[])
        self.axLoss.plot([],[])
        self.axLoss.set_xlabel('Epoch')
        self.axLoss.set_ylabel('Loss')
    
    def UpdatePlots(self, epoch, loss, validation):
        """Update the loss plots with new data.

        Parameters:
            epoch (int): Current epoch number.
            loss (float): Training loss.
            validation (float): Validation loss.
        """
        currentplot = self.axLoss.get_lines()
        currentplotLossx = currentplot[0].get_xdata()
        currentplotLossy = currentplot[0].get_ydata()
        currentplotVLossx = currentplot[1].get_xdata()
        currentplotVLossy = currentplot[1].get_ydata()

        # Remove old plots and add new data
        currentplot[1].remove()
        currentplot[0].remove()
        self.axLoss.plot(np.append(currentplotLossx,int(epoch)),
                        np.append(currentplotLossy, loss),'b-')

        self.axLoss.plot(np.append(currentplotVLossx,int(epoch)),
                        np.append(currentplotVLossy, validation),'r-')

        self.axLoss.set_xlabel('Epoch')
        self.axLoss.set_ylabel('Loss')
        
        
        # Update the display
        self.hdisplay.update(self.figLoss)
