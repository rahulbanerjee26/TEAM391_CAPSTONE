import numpy as np
import device
import model as ppg_model
import matplotlib.pyplot as plt
import pickle
from model import Classifier 

macAddress = "/dev/tty.BITalino-DevB"
training_signal = []
model1 = False
model2 = False
model3 = True


multipleFinger = False

if multipleFinger:
    input("Finger 1")
    training_signal.extend(device.collect_data(macAddress,run_time=20))
    input("Finger 2")
    training_signal.extend(device.collect_data(macAddress,run_time=20))
    input("Finger 3")
    training_signal.extend(device.collect_data(macAddress,run_time=20))
    input("Finger 4")
    training_signal.extend(device.collect_data(macAddress,run_time=20))
    input("Finger 5")
    training_signal.extend(device.collect_data(macAddress,run_time=10))
else:
    training_signal = device.collect_data(macAddress,run_time=90)

training_signal = np.array(training_signal)

cleaned_ppg_data_training = device.clean_data(training_signal, doPlot=False)

# Authentication
data = np.load("preprocessed_data.npy", allow_pickle=True)
_, _, authenticator_data = ppg_model.create_dataset(data)

participant = 24

# xxxxxxxxxxxxxxx model 1 xxxxxxxxxxxxxxxxxxxx
if model1:
    print("------  STARTED TRAINING 1")
    classifier_model = pickle.load(open('./Works_Classifier_Model.sav', 'rb'))
    train_accs, val_accs, test_accs, confusion_matrices , trained_model_1= ppg_model.train_model_for_participant(ppg_model.Authenticator, classifier_model, authenticator_data, cleaned_ppg_data_training ,participant)

    mean = lambda x: sum(x)/len(x)

    print("Average final train accuracy: {}".format(mean(train_accs)))
    print("Average final validation accuracy: {}".format(mean(val_accs)))
    print("Average test accuracy: {}".format(mean(test_accs)))
    print(sum(confusion_matrices))

    print("------  DONE TRAINING 1")

# xxxxxxxxxxxxxxx model 2 xxxxxxxxxxxxxxxxxxxx
if model2:
    print("------  STARTED TRAINING 2")
    classifier_model = pickle.load(open('./1_Classifier_Model.sav', 'rb'))
    train_accs, val_accs, test_accs, confusion_matrices , trained_model_2= ppg_model.train_model_for_participant(ppg_model.Authenticator, classifier_model, authenticator_data, cleaned_ppg_data_training ,participant)

    mean = lambda x: sum(x)/len(x)

    print("Average final train accuracy: {}".format(mean(train_accs)))
    print("Average final validation accuracy: {}".format(mean(val_accs)))
    print("Average test accuracy: {}".format(mean(test_accs)))
    print(sum(confusion_matrices))

    print("------  DONE TRAINING 2")

# xxxxxxxxxxxxxxx model 3 xxxxxxxxxxxxxxxxxxxx
if model3:
    print("------  STARTED TRAINING 3")
    classifier_model = pickle.load(open('./2_Classifier_Model.sav', 'rb'))
    train_accs, val_accs, test_accs, confusion_matrices , trained_model_3= ppg_model.train_model_for_participant(ppg_model.Authenticator, classifier_model, authenticator_data, cleaned_ppg_data_training ,participant)

    mean = lambda x: sum(x)/len(x)

    print("Average final train accuracy: {}".format(mean(train_accs)))
    print("Average final validation accuracy: {}".format(mean(val_accs)))
    print("Average test accuracy: {}".format(mean(test_accs)))
    print(sum(confusion_matrices))

    print("------  DONE TRAINING 3")

while True:
    key = input("New Preiction? Y for Yes, Q to quit:  ")
    if key == 'q' or key == 'Q':
        print("Trademark of Dimitrios Hatzinokos")
        break
    elif key == 'y':
        try:
            test_signal = device.collect_data(macAddress,run_time=10)
            test_signal = np.array(test_signal)
            cleaned_test_signal= device.clean_data(test_signal, doPlot=False)
            if model1:
                model_pred1 = ppg_model.prediction(data=cleaned_test_signal, model=trained_model_1)
                print("MODEL 1: ",model_pred1.argmax(dim=1))
            if model2:
                model_pred2 = ppg_model.prediction(data=cleaned_test_signal, model=trained_model_2)
                print("MODEL 2: ",model_pred2.argmax(dim=1))
            if model3:
                model_pred3 = ppg_model.prediction(data=cleaned_test_signal, model=trained_model_3)
                print("MODEL 3: ",model_pred3.argmax(dim=1))         
        except:
            print("error occured")


    elif key == 's':
        try:
            test_signal = device.collect_data(macAddress,run_time=20)
            test_signal = np.array(test_signal)
            cleaned_test_signal= device.clean_data(test_signal, doPlot=False)
            if model1:
                model_pred1 = ppg_model.prediction(data=cleaned_test_signal, model=trained_model_1)
                print("MODEL 1: ",model_pred1.argmax(dim=1))
            if model2:
                model_pred2 = ppg_model.prediction(data=cleaned_test_signal, model=trained_model_2)
                print("MODEL 2: ",model_pred2.argmax(dim=1))
            if model3:
                model_pred3 = ppg_model.prediction(data=cleaned_test_signal, model=trained_model_3)
                print("MODEL 3: ",model_pred3.argmax(dim=1))    
        except:
            print("error occured")

    elif key == 'r':
        try:
            test_signal = device.collect_data(macAddress,run_time=10)
            test_signal = np.array(test_signal)
            cleaned_test_signal= device.clean_data(test_signal, doPlot=False, filter=False)
            if model1:
                model_pred1 = ppg_model.prediction(data=cleaned_test_signal, model=trained_model_1)
                print("MODEL 1: ",model_pred1.argmax(dim=1))
            if model2:
                model_pred2 = ppg_model.prediction(data=cleaned_test_signal, model=trained_model_2)
                print("MODEL 2: ",model_pred2.argmax(dim=1))
            if model3:
                model_pred3 = ppg_model.prediction(data=cleaned_test_signal, model=trained_model_3)
                print("MODEL 3: ",model_pred3.argmax(dim=1))    
        except:
            print("error occured")

