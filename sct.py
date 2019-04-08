import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("sct_data.csv") 

day = data["What Day is it today?"]

fresh = data["What was the condition of the vegetables before cooking?"]

animals = data["Is the mess free from animals like cats and dogs?"] 

expirence = data["How experienced is the mess worker who is cooking the food?"]

peeled = data["Were the vegetables washed and peeled properly?"]

cleanliness = data["How would you rate the overall cleanliness of the mess?"]

roti_wastage = data["Roti / Paratha wastage (food left over in the plates)"]

dal_wastage = data["Dal wastage (food left over in the plates)"]

sabji_wastage = data["Sabji wastage (food left over in the plates)"]

rice_wastage = data["Rice Wastage (food left over in the plates)"]

#print (rice_wastage)
day_crisp = np.zeros((day.size, 7), dtype=int)

fresh_fuzzy = np.zeros((fresh.size, 3),  dtype=int)

animals_crisp = np.zeros(animals.size, dtype=int)


peeled = np.zeros((peeled.size, 3) ,dtype=int)

cleanliness_fuzzy = np.zeros((cleanliness.size, 3), dtype=float)

day_arr = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_bin = [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]

day_crisp = [day_bin[day_arr.index(i)] for i in day]

fresh_mem = [[-1,-1,1], [-1,1,1]]

fresh_fuzzy = [fresh_mem[1] if i == "Fresh" else fresh_mem[0] for i in fresh ]

animals_crisp = [[-1] if i == "Yes" else [1] for i in animals]

peeled_fuzzy = [fresh_mem[0] if i == "Yes" else fresh_mem[1] for i in peeled]

cleanliness_mem = [[-1,-1,0], [-0.5,0,0.5], [0,1,1]]

clean_arr = ["Dirty", "Average", "Excellent"]

fresh_fuzzy = [fresh_mem[0] if i == "Fresh" else fresh_mem[1] for i in fresh ]

cleanliness_fuzzy =  [cleanliness_mem[clean_arr.index(i)] for i in cleanliness]

experience_mem = [[-1,-1,-0.83333, -0.5], [-0.83333, -0.5, -.1666, .1666], [-.1666, .1666, 0.5, 0.83333], [0.5, .8333, 1,1]]

exp_arr = ["<1 Year", "1-3 Years", "3-5 Years", ">5 Years"]

experience_fuzzy =  [experience_mem[exp_arr.index(i)] for i in expirence]

final_data = [day_crisp[i]+fresh_fuzzy[i]+animals_crisp[i]+peeled_fuzzy[i]+cleanliness_fuzzy[i]+experience_fuzzy[i] for i in range(56)]
final_data = np.array(final_data)

#wastage_mem = [[-1,-1,0.5], [-1,-0.5,0], [-0.5,0 0.5], [0,0.5, 1], [0.5, 1,1]]
wastage = [[-1], [-0.5], [0], [0.5], [1]]

wastage_arr = [1,2,3,4,5]

roti_wastage_output = [wastage[wastage_arr.index(i)] for i in roti_wastage]

sabji_wastage_output = [wastage[wastage_arr.index(i)] for i in sabji_wastage]

dal_wastage_output = [wastage[wastage_arr.index(i)] for i in dal_wastage]

rice_wastage_output = [wastage[wastage_arr.index(i)] for i in rice_wastage] 

wastage_output = [roti_wastage_output[i] + dal_wastage_output[i] + sabji_wastage_output[i] + rice_wastage_output[i] for i in range(56)]
wastage_output = np.array(wastage_output)

input_size = 21
hidden_size = 40
num_classes = 4
learning_rate = 0.001
num_epochs = 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)#.to(device)
final_data = torch.from_numpy(final_data)
wastage_output = torch.from_numpy(wastage_output)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

total_step = 56
for epoch in range(num_epochs):
    	#for i, (final_data, wastage_output) in enumerate(final_data):
        #final_data = final_data.to(device)
        #wastage_output = wastage_output.to(device)
        outputs = model(final_data)
        loss = criterion(outputs, wastage_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if (i+1) % 10 == 0:
        print ('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, 1, total_step, loss.item()))

with torch.no_grad():
    correct = 0 
    total = 0
    for final_data, wastage_output in (test_loader):
        final_data = final_data.to(device)
        wastage_output = wastage_output.to(device)
        outputs = model(final_data)
        _, predicted = torch.max(outputs.data, 1)
        total+=predicted.size(0)
        correct += (predicted==wastage_output).sum()
        
    print('Accuracy of the network on the Training set: {} %'.format(100 * correct / total))

