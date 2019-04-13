from django.shortcuts import render, redirect, HttpResponse
import csv
import os
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
Database =  'Database.csv'

Name = None 
Roll_No = None
Hall = None 
Weekday = None 
Meal = None
Freshness = None
Experience = None
Washed_Peeled = None
Animals = None
Cleanliness = None
comments = None
quality = None
Infer_model = "/home/yash/Soft_Comm/Model_CQ"

def index(request):
    if 'count' not in request.session:
        request.session['count'] = 0
    request.session['count'] += 1
    return render(request, "surveyApp/index.html")

def create_survey(request):
    row = []
    print("''''''''''kjbsbsb''''''''''")
    if request.method == "POST":
        Name = request.session['Name'] = request.POST['Name']
        row.append(Name)
        Roll_No = request.session['Roll_No'] = request.POST['Roll_No']
        row.append(Roll_No)
        Hall = request.POST['Hall']
        row.append(Hall)
        Weekday = request.POST['Weekday']
        row.append(Weekday)
        Meal = request.POST['Meal']
        row.append(Meal)
        Freshness = request.POST['Freshness']
        row.append(Freshness)
        Experience = request.POST['Experience']
        row.append(Experience)
        # Washed_Peeled = request.POST['Washed_Peeled']
        # row.append(Washed_Peeled)
        Animals = request.POST['Animals']
        row.append(Animals)
        Cleanliness = request.POST['Cleanliness']
        row.append(Cleanliness)
        comments = request.POST['comments']
        row.append(comments)
        quality = request.POST['quality']
        row.append(quality)
        with open(os.path.abspath(Database), 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        print(row)
        Waste = infer(Infer_model)[0] 
        
        if Waste > 5:
            Waste = 5
        elif Waste < 0:
            Waste = 0

        if Waste < 2.25:
            Waste_fuzzy = "Low"
        elif Waste < 3.75:
            Waste_fuzzy = "Average"
        else:
            Waste_fuzzy = "High"

        request.session['Waste'] = Waste_fuzzy+"("+str(Waste)+")"#float(Waste[0]) 
        print(row)
        #request.session['locations'] = request.POST['locations']
        #request.session['languages'] = request.POST['languages']
        #request.session['comments'] = request.POST['comments']
        return redirect('/result')
    

def submitted_info(request):
    return render (request, 'surveyApp/result.html')

def infer(Model):
    model = torch.load(Model)
    model.eval()
    inp_d = get_Data(Database)[-1]
    with torch.no_grad():
        inp_d = torch.from_numpy(inp_d)
        outputs = model(inp_d)
        outputs = np.array(outputs)

    return outputs

def get_Data(file_h):
    data = pd.read_csv(file_h) 

    day = data["Weekday"]

    fresh = data["Freshness"]

    animals = data["Animals"] 

    expirence = data["Experience"]

    # peeled = data["Washed_Peeled"]

    cleanliness = data["Cleanliness"]

    
    comments = data["comments"]

    quality = data["quality"]

    #print (rice_wastage)
    day_crisp = np.zeros((day.size, 7), dtype=int)

    fresh_fuzzy = np.zeros((fresh.size, 3),  dtype=float)

    animals_crisp = np.zeros(animals.size, dtype=int)


    # peeled_fuzzy = np.zeros((peeled.size, 3) ,dtype=float)
    experience_fuzzy = np.zeros((expirence.size, 4) ,dtype=float)

    cleanliness_fuzzy = np.zeros((cleanliness.size, 3), dtype=float)

   
    fresh_mem = [[-1,-1,1], [-1,1,1]]

    fresh_fuzzy = [fresh_mem[1] if i == "Fresh" else fresh_mem[0] for i in fresh ]

    animals_crisp = [[-1] if i == "Yes" else [1] for i in animals]

    comments_crisp = [[1] if "Bad Taste" in i else [-1] for i in comments]

    # peeled_fuzzy = [fresh_mem[0] if i == "Yes" else fresh_mem[1] for i in peeled]

    cleanliness_mem = [[-1,-1,0], [-0.5,0,0.5], [0,1,1]]

    clean_arr = ["Dirty", "Average", "Excellent"]

    
    cleanliness_fuzzy =  [cleanliness_mem[clean_arr.index(i)] for i in cleanliness]

    experience_mem = [[-1,-1,-0.83333, -0.5], [-0.83333, -0.5, -.1666, .1666], [-.1666, .1666, 0.5, 0.83333], [0.5, .8333, 1,1]]

    exp_arr = ["1 Year", "1-3 Year", "3-5 Year", "5 Year"]

    experience_fuzzy =  [experience_mem[exp_arr.index(i)] for i in expirence]
    qual_mem = [[-1,-1,-0.5], [-1,-0.5,0], [-0.5,0 ,0.5], [0,0.5, 1], [0.5, 1,1]]

    wastage_arr = [1,2,3,4,5]

    qual_fuzzy = [qual_mem[wastage_arr.index(i)] for i in quality]

    final_data = [comments_crisp[i]+fresh_fuzzy[i]+animals_crisp[i]+cleanliness_fuzzy[i]+qual_fuzzy[i] + experience_fuzzy[i]  for i in range(len(animals_crisp))]

    rng_state = np.random.get_state()
    final_data = np.array(final_data)
    final_data = final_data.astype('float32')
    print (final_data.shape)
    return final_data