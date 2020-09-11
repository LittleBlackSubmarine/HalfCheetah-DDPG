import matplotlib.pyplot as plt
import numpy as np


# Mean value filter
k=2
kern=np.ones(2*k+1)/(2*k+1)


f1 = open("128_64.csv", "r")
f2 = open("400_300.csv", "r")
f3 = open("64_64.csv", "r")
f4 = open("32_32.csv", "r")
f5 = open("200_200.csv", "r")

distance1 = []
distance2 = []
distance3 = []
distance4 = []
distance5 = []
episodes = []


for d1, d2, d3, d4, d5 in zip(f1, f2, f3, f4, f5):
    distance1.append(int(d1[0:-1]))
    distance2.append(int(d2[0:-1]))
    distance3.append(int(d3[0:-1]))
    distance4.append(int(d4[0:-1]))
    distance5.append(int(d5[0:-1]))


for i in range (2500):
    episodes.append(i)
    i += 1

# Filtering signals
distance1=np.convolve(distance1,kern, mode='same')
distance2=np.convolve(distance2,kern, mode='same')
distance3=np.convolve(distance3,kern, mode='same')
distance4=np.convolve(distance4,kern, mode='same')
distance5=np.convolve(distance5,kern, mode='same')

# Plotting results
plt.figure()
plt.title("Distance over episodes")
plt.xlabel("Episodes"), plt.ylabel("Distance [cm]")
plt.axis([0,2500,-500,2500])
plt.plot(episodes, distance1, "orange", label = "128_64")
plt.plot(episodes, distance2, "r", label = "400_300")
plt.plot(episodes, distance3, "b", label = "64_64")
plt.plot(episodes, distance4, "c", label = "32_32")
plt.plot(episodes, distance5, "g", label = "200_200")
plt.legend()
plt.show()