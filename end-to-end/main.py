import numpy as np
import device
import model as ppg_model
import matplotlib.pyplot as plt
import pickle
from model import Classifier 
import streamlit as st
import torch

macAddress = "/dev/tty.BITalino-DevB"
training_signal = []


def collect_data(is_streamlit = False,run_time = 90):
    training_signal = device.collect_data(macAddress,run_time=run_time,is_streamlit=True)

    training_signal = np.array(training_signal)

    if not is_streamlit:
        cleaned_ppg_data_training = device.clean_data(training_signal, doPlot=False)
    else:
        with st.spinner('Cleaning Data.....'):
            cleaned_ppg_data_training = device.clean_data(training_signal, doPlot=False)
            if is_streamlit:
                st.success("Data Collection and Cleaning is Complete!")


    # Authentication
    data = np.load("preprocessed_data.npy", allow_pickle=True)
    _, _, authenticator_data = ppg_model.create_dataset(data)

    return authenticator_data,cleaned_ppg_data_training


def train_model(authenticator_data,cleaned_ppg_data_training,is_streamlit = False,model_name=''):
    participant = 24

    print("------  STARTED TRAINING")
    classifier_model = pickle.load(open('./2_Classifier_Model.sav', 'rb'))
    if not is_streamlit:
        train_accs, val_accs, test_accs, confusion_matrices , trained_model= ppg_model.train_model_for_participant(ppg_model.Authenticator, classifier_model, authenticator_data, cleaned_ppg_data_training ,participant)
    else:
        with st.spinner('Training Model.....'):
            train_accs, val_accs, test_accs, confusion_matrices , trained_model= ppg_model.train_model_for_participant(ppg_model.Authenticator, classifier_model, authenticator_data, cleaned_ppg_data_training ,participant)
            torch.save(trained_model,open(f'Authenticator_Models/{model_name}.sav', 'wb'))
        st.balloons()
        st.success("The Model has been trained on your PPG Data :)")

    mean = lambda x: sum(x)/len(x)

    print("Average final train accuracy: {}".format(mean(train_accs)))
    print("Average final validation accuracy: {}".format(mean(val_accs)))
    print("Average test accuracy: {}".format(mean(test_accs)))
    print(sum(confusion_matrices))
    print("------  DONE TRAINING")

    return trained_model


def predict(data,model,is_streamlit = False):
    model_pred = ppg_model.prediction(data=data, model=model)
    preds = model_pred.argmax(dim=1)
    print("MODEL 3: ",preds)

    if is_streamlit:
        if 1 in preds:
            st.success("Authenticated")
        else:
            st.error("Not Authenticated")   





