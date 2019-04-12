from django.shortcuts import render, redirect, HttpResponse
import csv
import os
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

def index(request):
    if 'count' not in request.session:
        request.session['count'] = 0
    request.session['count'] += 1
    return render(request, "surveyApp/index.html")

def create_survey(request):
    row = []
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
        Washed_Peeled = request.POST['Washed_Peeled']
        row.append(Washed_Peeled)
        Animals = request.POST['Animals']
        row.append(Animals)
        Cleanliness = request.POST['Cleanliness']
        row.append(Cleanliness)
        comments = request.POST['comments']
        row.append(comments)
        with open(os.path.abspath(Database), 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        #request.session['locations'] = request.POST['locations']
        #request.session['languages'] = request.POST['languages']
        #request.session['comments'] = request.POST['comments']
        return redirect('/result')
    else:
        return redirect('/')

def submitted_info(request):
    return render (request, 'surveyApp/result.html')
