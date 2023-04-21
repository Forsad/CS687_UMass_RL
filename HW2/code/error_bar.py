import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# if using a Jupyter notebook, include:

if __name__ == '__main__':
	#db = pd.read_csv('/home/forsad/Desktop/cs687/HW2/code/out.csv')
	db = pd.read_csv('out.csv')
	print(db.columns)
	trs = list(db['Tabular Random Search'])
	bbo = list(db['BBO'])
	trs_err = list(db['TRS Error Bar'])
	bbo_err = list(db['BBO Error Bar'])
	x = np.arange(len(trs))

	fig, ax = plt.subplots()

	ax.errorbar(x, trs,
	            yerr=trs_err,
	            fmt='-.', label='TRS')
	ax.errorbar(x, bbo, yerr=bbo_err, fmt='-.', label='BBO')

	ax.set_xlabel('x-axis')
	ax.set_ylabel('y-axis')
	ax.set_title('TRS vs BBO error bar')
	ax.legend()
	plt.show()