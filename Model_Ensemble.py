from Evaluation import evaluation
from Model_Mobilenet import MobileNet
from Model_VGG16 import Model_VGG16


def Model_Ensemble(Train_Data, Train_Target, Test_Data, Test_Target):
    out, Eval_mbl = MobileNet(Train_Data, Test_Target, Train_Data.shape[0], num_units=2)
    out, Eval_vgg = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target)

    pred = (Eval_mbl + Eval_vgg) / 2

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, Test_Target)
    return Eval, pred