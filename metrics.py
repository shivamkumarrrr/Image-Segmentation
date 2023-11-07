import sklearn.metrics
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils import data
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.optim as optim
from IPython.display import clear_output
import os
import pandas as pd
from IPython.display import HTML, display
# import tabulate
from tqdm import tqdm
    
class EvalResult:
    """
    Provides structure for storing the metrics
    """
    def __init__(self, dice, jaccard, confusion_matrix):
        super(EvalResult, self).__init__()
        self.dice = dice
        self.jaccard = jaccard
        self.confusion_matrix = confusion_matrix

def evaluate(model, dataset, device):
    """
    Evaluates the model with respect to the dataset.
    
    returns: 
    EvalResult object with Dice Coefficient,Jaccard Similarity, and the Confusion Matrix
    
    input: 
    model - model to be evaluated of type nn.Module.
    dataset - dataset object in nn.Dataset format.

    """
    size = 19
    dice = np.zeros(size)
    dice_count = np.zeros(size)
    jaccard = np.zeros(size)
    jaccard_count = np.zeros(size)
    confusion_matrix = np.zeros((size, size))

    trainloader = data.DataLoader(dataset, batch_size = 1, num_workers = 10)

    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(trainloader), total=len(dataset)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            outputs = outputs.cpu()
            targets = targets.cpu()


            targetValues, targetIndices = targets.max(dim=1)
            values, indices = outputs.max(dim=1)

            excluded = targetValues < 0.1
            targetIndices[excluded] = size
            indices[excluded] = size

            flattenTarget = targetIndices.flatten()
            flattenOutput = indices.flatten()

            confusion_matrix += sklearn.metrics.confusion_matrix(flattenTarget, flattenOutput, labels=range(size))

            for k in range(size):
                targetIsClass = flattenTarget==k
                outputIsClass = flattenOutput==k
                # calculate dice score
                occured = (torch.sum(outputIsClass) + torch.sum(targetIsClass))

                #if denominator is 0 => class not in image to be concidered
                if occured != 0:
                    dice_count[k] += 1
                    dice[k] += torch.sum(flattenOutput[targetIsClass]==k)*2.0 / occured

                denominator = torch.sum(torch.logical_or(targetIsClass, outputIsClass))

                #if denominator is 0 => class not in image to be concidered
                if denominator != 0:
                    jaccard_count[k] += 1
                    jaccard[k] += torch.sum(torch.logical_and(targetIsClass, outputIsClass)) / denominator


    dice /= dice_count 
    jaccard /= jaccard_count 
    
    return EvalResult(dice, jaccard, confusion_matrix)

# def plot(table):
#     """
#     Plots the evaluation results in a clean tabular form on the jupyter notebook. - deprecated.
#     """
#     display(HTML(tabulate.tabulate(torch.tensor(table), tablefmt='html')))

def displayEval(evalResult):
    """
    Displays the evaluation results in a clean tabular format in the Jupyter Notebook.

    input: object of type EvalResult.
    """
    
    confusion_matrix = evalResult.confusion_matrix
    stripped_dice = evalResult.dice
    stripped_jaccard = evalResult.jaccard

    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    SE = np.nan_to_num(TP/(TP+FN))
    SP = np.nan_to_num(TN/(TN+FP))
    PC = np.nan_to_num(TP/(TP + FP))
    F1 = np.nan_to_num(2 * (PC * SE) / (PC + SE))
    ACC = (TP+TN)/(TP+FP+FN+TN)


    names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    frame = [np.append(names, "**Average"), 
             np.append(SE, np.sum(SE) / 19), 
             np.append(SP, np.sum(SP) / 19), 
             np.append(ACC, np.sum(ACC) / 19), 
             np.append(F1, np.sum(F1) / 19), 
             np.append(stripped_dice, np.sum(stripped_dice) / 19), 
             np.append(stripped_jaccard, np.sum(stripped_jaccard) / 19)]


    display(pd.DataFrame(np.stack(frame, axis=1), columns=["Name","SE","SP","ACC","F1","Dice", "Jaccard"]))
