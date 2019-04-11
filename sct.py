import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later)  3.375 azad 2.333 hjb 1.88 nehru
def create_data(file_h):
    data = pd.read_csv(file_h) 

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
    comments = data["Why do you think there will be food wastage in your mess today?"]

    #print (rice_wastage)
    day_crisp = np.zeros((day.size, 7), dtype=int)

    fresh_fuzzy = np.zeros((fresh.size, 3),  dtype=float)

    animals_crisp = np.zeros(animals.size, dtype=int)


    peeled_fuzzy = np.zeros((peeled.size, 3) ,dtype=float)
    experience_fuzzy = np.zeros((peeled.size, 4) ,dtype=float)

    cleanliness_fuzzy = np.zeros((cleanliness.size, 3), dtype=float)

    # day_arr = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # day_bin = [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]

    # day_crisp = [day_bin[day_arr.index(i)] for i in day]

    fresh_mem = [[-1,-1,1], [-1,1,1]]

    fresh_fuzzy = [fresh_mem[1] if i == "Fresh" else fresh_mem[0] for i in fresh ]

    animals_crisp = [[-1] if i == "Yes" else [1] for i in animals]

    comments_crisp = [[1] if "Bad Taste" in i else [-1] for i in comments]

    peeled_fuzzy = [fresh_mem[0] if i == "Yes" else fresh_mem[1] for i in peeled]

    cleanliness_mem = [[-1,-1,0], [-0.5,0,0.5], [0,1,1]]

    clean_arr = ["Dirty", "Average", "Excellent"]

    fresh_fuzzy = [fresh_mem[0] if i == "Fresh" else fresh_mem[1] for i in fresh ]

    cleanliness_fuzzy =  [cleanliness_mem[clean_arr.index(i)] for i in cleanliness]

    experience_mem = [[-1,-1,-0.83333, -0.5], [-0.83333, -0.5, -.1666, .1666], [-.1666, .1666, 0.5, 0.83333], [0.5, .8333, 1,1]]

    exp_arr = ["<1 Year", "1-3 Years", "3-5 Years", ">5 Years"]

    experience_fuzzy =  [experience_mem[exp_arr.index(i)] for i in expirence]
    print(type(day_crisp))
    final_data = [comments_crisp[i] + fresh_fuzzy[i]+animals_crisp[i]+peeled_fuzzy[i]+cleanliness_fuzzy[i]+experience_fuzzy[i] for i in range(len(animals_crisp))]

    rng_state = np.random.get_state()
    final_data = np.array(final_data)
    # np.random.shuffle(final_data)

    #wastage_mem = [[-1,-1,0.5], [-1,-0.5,0], [-0.5,0 0.5], [0,0.5, 1], [0.5, 1,1]]
    wastage = [[1], [2], [3], [4], [5]]

    wastage_arr = [1,2,3,4,5]

    roti_wastage_output = np.array([wastage[wastage_arr.index(i)] for i in roti_wastage])

    sabji_wastage_output = np.array([wastage[wastage_arr.index(i)] for i in sabji_wastage])

    dal_wastage_output = np.array([wastage[wastage_arr.index(i)] for i in dal_wastage])

    rice_wastage_output = np.array([wastage[wastage_arr.index(i)] for i in rice_wastage]) 
    print(roti_wastage_output)
    wastage_output = np.add(0.4*roti_wastage_output, 0.6*sabji_wastage_output)#, np.add(0.1*dal_wastage_output, 0.1*rice_wastage_output)) #[roti_wastage_output[i] + sabji_wastage_output[i] + dal_wastage_output[i] + rice_wastage_output[i]  for i in range(7)]
    
    return final_data, wastage_output

final_data, wastage_output = create_data("sct_data.csv")
final_data_2, wastage_output_2 = create_data("sct_data2.csv")


final_data_2 = np.array(final_data_2)




wastage_output_2 = np.array(wastage_output_2)

input_size = 15
hidden_size = [40, 8]
num_classes = 1
learning_rate = 0.001
num_epochs = 4000

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        # self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[0], num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
nu = 50
model = torch.load("Model")#NeuralNet(input_size, hidden_size, num_classes)#.to(device)
model.eval()
final_data = final_data.astype('float32')

final_data_2 = final_data_2.astype('float32')
final_data_train = torch.from_numpy(final_data)
final_data_test = torch.from_numpy(final_data_2)

wastage_output = wastage_output.astype('float32')
wastage_output_2 = wastage_output_2.astype('float32')

wastage_output_train = torch.from_numpy(wastage_output)
wastage_output_test = torch.from_numpy(wastage_output_2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

total_step = 56

# for epoch in range(num_epochs):
#     	#for i, (final_data, wastage_output) in enumerate(final_data):
#         #final_data = final_data.to(device)
#         #wastage_output = wastage_output.to(device)
#         # print(final_data.shape)
#         outputs = model(final_data_train)
#         loss = criterion(outputs, wastage_output_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         #if (i+1) % 10 == 0:
#         print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
# torch.save(model, "Model")
with torch.no_grad():
    correct = 0 
    total = 0
    outputs = model(final_data_train)
    outputs = np.array(outputs)
    
    t = np.linspace(0, 2*outputs.shape[0], outputs.shape[0])
    a = outputs
    b = np.array(wastage_output_train)

    # a1 = outputs[:,1]
    # b1 = np.array(wastage_output_train)[:,1]

    # a2 = outputs[:,2]
    # b2 = np.array(wastage_output_train)[:,2]

    # a3 = outputs[:,3]
    # b3 = np.array(wastage_output_train)[:,3]

# b = cos(t)

plt.subplot(4,1,1)
plt.scatter(t, a) # plotting t, a separately 
plt.scatter(t, b,s=2,c="red") # plotting t, b separately 
plt.plot(t, b, c="red") # plotting t, c separately 
print("CLEARED")
# plt.subplot(4,1,2)
# plt.scatter(t, a1) # plotting t, a separately 
# plt.scatter(t, b1,s=2,c="red") # plotting t, b separately 
# plt.plot(t, b1, c="red") # plotting t, c separately 

# plt.subplot(4,1,3)
# plt.scatter(t, a2) # plotting t, a separately 
# plt.scatter(t, b2,s=2,c="red") # plotting t, b separately 
# plt.plot(t, b2, c="red") # plotting t, c separately 

# plt.subplot(4,1,4)
# plt.scatter(t, a3) # plotting t, a separately 
# plt.scatter(t, b3,s=2,c="red") # plotting t, b separately 
# plt.plot(t, b3, c="red") # plotting t, c separately 
# plt.show()

with torch.no_grad():
    correct = 0 
    total = 0
    outputs = model(final_data_test)
    outputs = np.array(outputs)
    print(outputs)
    t1 = np.linspace(0, 2*outputs.shape[0], outputs.shape[0])
    a1 = outputs
    b1 = np.array(wastage_output_test)

    # a1 = outputs[:,1]
    # b1 = np.array(wastage_output_test)[:,1]

    # a2 = outputs[:,2]
    # b2 = np.array(wastage_output_test)[:,2]

    # a3 = outputs[:,3]
    # b3 = np.array(wastage_output_test)[:,3]

# b = cos(t)

plt.subplot(4,1,2)
plt.scatter(t1, a1) # plotting t, a separately 
plt.scatter(t1, b1,s=2,c="red") # plotting t, b separately 
plt.plot(t1, b1, c="red") # plotting t, c separately 

# plt.subplot(4,1,2)
# plt.scatter(t, a1) # plotting t, a separately 
# plt.scatter(t, b1,s=2,c="red") # plotting t, b separately 
# plt.plot(t, b1, c="red") # plotting t, c separately 

# plt.subplot(4,1,3)
# plt.scatter(t, a2) # plotting t, a separately 
# plt.scatter(t, b2,s=2,c="red") # plotting t, b separately 
# plt.plot(t, b2, c="red") # plotting t, c separately 

# plt.subplot(4,1,4)
# plt.scatter(t, a3) # plotting t, a separately 
# plt.scatter(t, b3,s=2,c="red") # plotting t, b separately 
# plt.plot(t, b3, c="red") # plotting t, c separately 
plt.show()

        
   