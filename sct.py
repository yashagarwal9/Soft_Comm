import pandas as pd 
import numpy as np
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

print day.size
day_crisp = np.zeros((day.size, 7), dtype=int)

fresh_fuzzy = np.zeros((fresh.size, 3),  dtype=int)

animals_crisp = np.zeros(animals.size, dtype=int)


peeled = np.zeros((peeled.size, 3) ,dtype=int)

cleanliness_fuzzy = np.zeros((cleanliness.size, 3), dtype=int)

day_arr = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_bin = [[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]

day_crisp = [day_bin[day_arr.index(i)] for i in day]

fresh_mem = [[-1,-1,1], [-1,1,1]]

fresh_fuzzy = [fresh_mem[1] if i == "Fresh" else fresh_mem[0] for i in fresh ]

animals_crisp = [-1 if i == "Yes" else 1 for i in animals]

peeled_fuzzy = [fresh_mem[0] if i == "Yes" else fresh_mem[1] for i in peeled]

cleanliness_mem = [[-1,-1,0], [-0.5,0,0.5], [0,1,1]]

clean_arr = ["Dirty", "Average", "Excellent"]

fresh_fuzzy = [fresh_mem[0] if i == "Fresh" else fresh_mem[1] for i in fresh ]



cleanliness_fuzzy =  [cleanliness_mem[clean_arr.index(i)] for i in cleanliness]



experience_mem = [[-1,-1,-0.83333, -0.5], [-0.83333, -0.5, -.1666, .1666], [-.1666, .1666, 0.5, 0.83333], [0.5, .8333, 1,1]]

exp_arr = ["<1 Year", "1-3 Years", "3-5 Years", ">5 Years"]


experience_fuzzy =  [experience_mem[exp_arr.index(i)] for i in expirence]

