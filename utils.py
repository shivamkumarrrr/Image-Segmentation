# +
import os
from PIL import Image
import torch
from torch.utils import data

from torchvision import transforms


# -

def removesuffix(content, suffix):
    """
    removes the suffix attached to the file name
    """
    if content.endswith(suffix):
        content = content[:-len(suffix)]
    return content

def findFiles(rootDir, suffix):
    """
    returns: list of files in the directory.
    """
    files = []
    for r, d, f in os.walk(rootDir):
        for file in f:
            if suffix in file:
                files.append(removesuffix(str(file), suffix))
    return files

def substringBefore(string, char):
    """
    returns the substring before a given character in a string
    """
    return string[:string.index(char)]

def createIfNotExist(directory):
    """
    create a directory if doesn't exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def resizeAllImages(src, target):
    """
    Preprocesses the data by resizing and saving it in the target path.

    input:
    src - source string where the leftImg8bit and gtFine packages of cityscapes dataset can be found.
    target - target path where the processed images has to be stored.
    """
    for dataset in ["train", "val"]:
        inputRoot = src + "/leftImg8bit/" + dataset + "/"
        targetRoot = src + "/gtFine/" + dataset + "/"
        inputSuffix = '_leftImg8bit.png'
        targetSuffix = '_gtFine_labelIds.png'

        reducedRoot = target + dataset + "/"

        createIfNotExist(reducedRoot + "input/")
        createIfNotExist(reducedRoot + "target/")

        files = findFiles(inputRoot, inputSuffix)
        #targetImages = findFiles(inputRoot, targetSuffix)

        i = 0;

        targetSize = (512, 256)

        for file in files:
            cityName = substringBefore(file, "_")

            input = Image.open(inputRoot + cityName + "/" + file + inputSuffix)
            target = Image.open(targetRoot + cityName + "/" + file + targetSuffix)

            input = input.resize(targetSize)
            target = target.resize(targetSize, Image.NEAREST)

            input.save(reducedRoot + "input/" + str(i).zfill(4) +  ".png")
            target.save(reducedRoot + "target/" + str(i).zfill(4) +  ".png")

            i += 1


def saveModel(model, root, epoch):
    """
    saves the model on the given path - deprecated.

    model - model to be saved.
    root- string indicating the path where the model will be saved.
    epoch - the epoch number to be appended to the model name.
    """
    torch.save(model, root + "/r2u_epoch_" + str(epoch) + ".model")

from IPython.display import clear_output,display
def displayTensorAsImage( tensor ):
    """
    display the given tensor as image on the Jupyter Notebook
    """
    display(transforms.ToPILImage()( tensor ))


def preview(inputs, outputs, targets, epoch):
    """
    Shows the preview of the images.

    inputs - input tensor that has to be displayed as image.
    outputs - output of the model
    targets - target tensors that has to be displayed as image.
    """

    colors = cityscapeColors()
    
    outputs = outputs.cpu()
    targets = targets.cpu()

    values, indices = outputs.max(dim=1)
    targetValues, targetIndices = targets.max(dim=1)

    #set areas with classes that are not be trained on to "unknown" class
    excluded = targetValues < 0.1
    targetIndices[excluded] = 19
    indices[excluded] = 19

    colorImage = torch.stack([colors[indices, 0 ], colors[indices, 1], colors[indices, 2]], dim=1)
    targetImage = torch.stack([colors[targetIndices, 0 ], colors[targetIndices, 1], colors[targetIndices, 2]], dim=1)

    clear_output(wait=False)
    print(epoch)
    displayTensorAsImage( inputs[0] )
    displayTensorAsImage( targetImage[0] )
    displayTensorAsImage( colorImage[0] )


# +
_colors = torch.tensor([
    [128, 64,128],
    [244, 35,232],
    [ 70, 70, 70],
    [102,102,156],
    [190,153,153],
    [153,153,153],
    [250,170, 30],
    [220,220,  0],
    [107,142, 35],
    [152,251,152],
    [ 70,130,180],
    [220, 20, 60],
    [255,  0,  0],
    [  0,  0,142],
    [  0,  0, 70],
    [  0, 60,100],
    [  0, 80,100],
    [  0,  0,230],
    [119, 11, 32],
    [0,0,0]
]) / 255

def cityscapeColors():
    return _colors
# -


def generateImages(model, dataset, device, root, images = range(20)):
    createIfNotExist(root)
    
    trainloader = data.DataLoader(
        dataset, 
        batch_size = 1, 
        num_workers = 10)

    pos = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(trainloader):
            if images[pos] < i:
                continue

            pos+=1

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            outputs = outputs.cpu()
            targets = targets.cpu()

            targetValues, targetIndices = targets.max(dim=1)
            values, indices = outputs.max(dim=1)

            noValidate = targetValues < 0.1
            targetIndices[noValidate] = 19
            indices[noValidate] = 19


            colorImage = torch.stack([_colors[indices, 0 ], _colors[indices, 1], _colors[indices, 2]], dim=1)
            targetImage = torch.stack([_colors[targetIndices, 0 ], _colors[targetIndices, 1], _colors[targetIndices, 2]], dim=1)

            targetImage = transforms.ToPILImage()( targetImage[0] )
            outputImage = transforms.ToPILImage()( colorImage[0] )

            display(targetImage)
            display(outputImage)

            targetImage.save(root + "/t2-gt-val" + str(pos) +  ".png")
            outputImage.save(root + "/t2-pred-val" + str(pos) +  ".png")

            if len(images) == pos:
                break
