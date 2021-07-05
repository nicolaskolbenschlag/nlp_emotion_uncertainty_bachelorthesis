import matplotlib.pyplot as plt

s1 = [1,1,1,1.5,2,2,2]
s2 = [2,2,2,1.5,1,1,1]

plt.plot(s1, c="r")
plt.plot(s2, c="b")

plt.ylim(0,3)

plt.axvline(x=3, ls="--", color="gray")

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel("Time", fontsize=18)
plt.ylabel("Emotional state", fontsize=18)

plt.show()
