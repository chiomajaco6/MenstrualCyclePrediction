from django.shortcuts import render
from django.http import HttpResponse
import joblib
import sklearn
import pickle
import numpy as np

# Create your views here.

def home(request):
    return render(request, 'home.html')

def result(request):
    # Load your trained model
    model = joblib.load("C:\\Users\\MINISTER JOHN\\Desktop\\OvulationDayPrediction\\DecisionTreeForOvulationDayPrediction.pkl")

    # Collect user inputs for the eleven features
    features = [
        request.GET['CycleNumber'],
        request.GET['LengthofCycle'],
        request.GET['LengthofLutealPhase'],
        request.GET['TotalNumberofHighDays'],
        request.GET['TotalNumberofPeakDays'],
        request.GET['UnusualBleeding'],
        request.GET['PhasesBleeding'],
        request.GET['IntercourseInFertileWindow'],
        request.GET['Age'],
        request.GET['BMI'],
        request.GET['Method'],
        
    ]



    # Make predictions using the model
    ans = model.predict([features])

    return render(request, 'result.html', {'ans': ans})
