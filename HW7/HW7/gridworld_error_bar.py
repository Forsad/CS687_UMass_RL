import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps
# if using a Jupyter notebook, include:

def plot_bar(ref_fl, csv_fl, typ, world, x_line):
	#db = pd.read_csv('/home/forsad/Desktop/cs687/HW2/code/out.csv')
	db = pd.read_csv(csv_fl)
	db_ref = pd.read_csv(ref_fl)
	reinforce = list(db['REINFORCE'])
	reinforce_ref = list(db_ref['REINFORCE'])
	reinforce_err = list(db['TRS Error Bar'])
	reinforce_ref_err = list(db_ref['TRS Error Bar'])
	x = np.arange(len(reinforce))

	fig, ax = plt.subplots()

	ax.errorbar(x, reinforce,
	            yerr=reinforce_err,
	            fmt='-.', label='REINFORCE')
	ax.errorbar(x, reinforce_ref, yerr=reinforce_ref_err, fmt='-.', label='REINFORCE reference')

	ax.set_xlabel('x-axis')
	ax.set_ylabel('y-axis')
	ax.set_title('REINFORCE vs REINFORCE reference error bar for ' + world + ": " + typ)
	ax.legend()

	print(typ + ": Area under curve for REINFORCE reference ", sum(reinforce_ref) / len(reinforce_ref_err))
	print(typ + ": Area under curve for REINFORCE ", sum(reinforce) / len(reinforce))
	print()

if __name__ == '__main__':
	try:
		plot_bar('out_ref.csv', 'out.csv', "algorithm", "Grid world", 0)
		plt.show()
	except KeyboardInterrupt:
		print("Ctrl+C pressed; Exiting...")