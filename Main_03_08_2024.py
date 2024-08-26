import numpy as np
import cv2 as cv
import os
import pandas as pd
from Learnper_Plot_Results import plot_results
from Model_CNN import Model_CNN
from Model_Ensemble import Model_Ensemble
from Model_Mobilenet import MobileNet
from Model_Resnet import Model_RESNET
from Model_VGG16 import Model_VGG16


def Read_Image(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (128, 128))
    return image


def Read_Images(Directory):
    Images = []
    out_folder = os.listdir(Directory)  
    for i in range(len(out_folder)):
        print(i)
        filename = Directory + out_folder[i]
        image = Read_Image(filename)
        Images.append(image)
    return Images


def Read_CSV(Path):
    df = pd.read_csv(Path)
    values = df.to_numpy()
    value = values[:, 2]
    uniq = np.unique(value)
    Target = np.zeros((len(value), len(uniq)))
    for i in range(len(uniq)):
        index = np.where(value == uniq[i])
        Target[index, i] = 1
    return Target



# Read Dataset

an = 0
if an == 1:
    Images = Read_Images('./Dataset/Images/')
    np.save('Images.npy', Images)

    Target = Read_CSV('./Dataset/HAM10000_metadata.csv')
    np.save('Target.npy', Target)


# Classification
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Learnper = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    Eval_all = []
    for i in range(len(Learnper)):
        learnperc = round(Images.shape[0] * Learnper[i])
        Train_Data = Images[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Images[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((5, 14))
        Eval[0, :], pred = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[1, :], pred = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[2, :], pred = MobileNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[3, :], pred = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[4, :], pred = Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval_all.append(Eval)
    np.save('Eval_all.npy', Eval_all)

plot_results()