import os
import joblib
import requests
import logging
from matplotlib import pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

