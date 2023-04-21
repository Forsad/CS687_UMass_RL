import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps
# if using a Jupyter notebook, include:

def plot_bar(csv_fl, typ, world, x_line):
	#db = pd.read_csv('/home/forsad/Desktop/cs687/HW2/code/out.csv')
	db = pd.read_csv(csv_fl)
	print(db.columns)
	sarsa = list(db['Sarsa'])
	qlearning = list(db['QLearning'])
	sarsa_err = list(db['Sarsa Error Bar'])
	qlearning_err = list(db['QLearning Error Bar'])
	x = np.arange(len(sarsa))

	fig, ax = plt.subplots()

	ax.errorbar(x, sarsa,
	            yerr=sarsa_err,
	            fmt='-.', label='SARSA')
	ax.errorbar(x, qlearning, yerr=qlearning_err, fmt='-.', label='Q-Learning')

	ax.set_xlabel('x-axis')
	ax.set_ylabel('y-axis')
	ax.set_title('SARSA vs Q-Learning error bar for ' + world + ": " + typ)
	ax.legend()
	sarsa_tmp = [x - x_line for x in sarsa]
	qlearning_tmp = [x - x_line for x in qlearning]
	print(typ + ": Area under curve for Sarsa ", simps(sarsa_tmp, dx=1))
	print(typ + ": Area under curve for QLearning ", simps(qlearning_tmp, dx=1))

if __name__ == '__main__':
	try:
		plot_bar('Gridworld_out_ref.csv', "reference", "Grid world", 0)
		plot_bar('Gridworld_out.csv', "algorithm", "Grid world", 0)
		plot_bar('Mountain Car_out_ref.csv', "reference", "Mountain Car", 0)
		plot_bar('Mountain Car_out.csv', "algorithm", "Mountain car", 0)
		plot_bar('Cart Pole_out_ref.csv', "Reference", "Cart Pole", 0)
		plot_bar('Cart Pole_out.csv', "Algorithm", "Cart Pole", 0)
		plt.show()
	except KeyboardInterrupt:
		print("Ctrl+C pressed; Exiting...")