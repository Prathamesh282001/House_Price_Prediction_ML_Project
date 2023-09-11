from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge

location_list = set(pd.read_pickle("./model/final_location_list.pkl"))
model = joblib.load('./model/Rmodel1.pkl')
x = pd.read_pickle('./model/final_df.pkl')

# Create your views here.

def home(request):
    context = {
                "location_list":location_list
                }
    return render(request,"index.html",context)

def predict(request):
    status = False
    if request.method == "POST":
        index = np.where(x.columns==request.POST.get('location'))[0][0]
        X = np.zeros(len(x.columns))
        X[0] = request.POST.get('area')
        X[1] = request.POST.get('bathroom')
        X[2] = request.POST.get('bedroom')
        if index >= 0:
            X[index] = 1

        status = True
        

    output = np.round(model.predict([X])[0][0],2)


    context = {'output':output,
                "status":status,
                "location_list":location_list}
    return render(request,'index.html',context)


"""
def predict(request):
	status = False
	print(request)
	if request.method == 'POST':
		dic={}
		dic['area']=request.POST.get('area')
		dic['location']=request.POST.get('location')
		dic['bedroom']=request.POST.get('bedroom')
		dic['bathroom']=request.POST.get('bathroom')
		
		status = True
		temp = dic.copy()
		print(dic.keys(),dic.values())
		

	df=pd.DataFrame({'input':dic}).transpose().astype("float")
	output = model.predict(df)[0]
	context = {'output':output,"temp":temp,"status":status,"location_list":location_list}
	return render(request,'home.html',context)
"""