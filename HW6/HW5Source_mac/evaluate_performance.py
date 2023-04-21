import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps
# if using a Jupyter notebook, include:
from random import randint
import os

def plot_bar(csv_fl, typ, world, x_line):
	#db = pd.read_csv('/home/forsad/Desktop/cs687/HW2/code/out.csv')
	db = pd.read_csv(csv_fl)
	print(db.columns)
	sarsa = list(db['Sarsa'])
	qlearning = list(db['QLearning'])
	sarsa_err = list(db['Sarsa Error Bar'])
	qlearning_err = list(db['QLearning Error Bar'])
	x = np.arange(len(sarsa))

	# fig, ax = plt.subplots()

	# ax.errorbar(x, sarsa,
	#             yerr=sarsa_err,
	#             fmt='-.', label='SARSA')
	# ax.errorbar(x, qlearning, yerr=qlearning_err, fmt='-.', label='Q-Learning')

	# ax.set_xlabel('x-axis')
	# ax.set_ylabel('y-axis')
	# ax.set_title('SARSA vs Q-Learning error bar for ' + world + ": " + typ)
	# ax.legend()
	sarsa_tmp = [x - x_line for x in sarsa]
	qlearning_tmp = [x - x_line for x in qlearning]
	areaSarsa = simps(sarsa_tmp, dx=1)
	areaQL = simps(qlearning_tmp, dx=1)
	print(typ + ": Area under curve for Sarsa ", areaSarsa)
	print(typ + ": Area under curve for QLearning ", areaQL)
	return (areaSarsa, areaQL)

def main():
	numRandSeeds = 50
	for _ in range(numRandSeeds):
		print("Staring ar new random seed trial")
		r1 = randint(0, 100000)
		r2 = randint(0, 100000)
		r3 = randint(0, 100000)
		cmd = "./run_system " + str(r1) + " " + str(r2) + " " + str(r3)
		print("Running command for seeds ", r1, ", ", r2, ", ", r3)
		os.system(cmd)
		(ref_gs, ref_gq) = plot_bar('Gridworld_out_ref.csv', "reference", "Grid world", 0)
		(alg_gs, alg_gq) = plot_bar('Gridworld_out.csv', "algorithm", "Grid world", 0)

		if alg_gs < ref_gs:
			raise Exception("Error for Gridworld AUC for SARSA: AUC_REF=", ref_gs, " ,AUC_ALG", alg_gs)

		if alg_gq < ref_gq:
			raise Exception("Error for Gridworld AUC for QLearning: AUC_REF=", ref_gq, " ,AUC_ALG", alg_gq)

		(ref_ms, ref_mq) = plot_bar('Mountain Car_out_ref.csv', "reference", "Mountain Car", 0)
		(alg_ms, alg_mq) = plot_bar('Mountain Car_out.csv', "algorithm", "Mountain car", 0)

		if alg_ms < ref_ms:
			raise Exception("Error for Mountain Car AUC for SARSA: AUC_REF=", ref_ms, " ,AUC_ALG", alg_ms)

		if alg_mq < ref_mq:
			raise Exception("Error for Mountain Car AUC for QLearning: AUC_REF=", ref_mq, " ,AUC_ALG", alg_mq)

		(ref_cs, ref_cq) = plot_bar('Cart Pole_out_ref.csv', "Reference", "Cart Pole", 0)
		(alg_cs, alg_cq) = plot_bar('Cart Pole_out.csv', "Algorithm", "Cart Pole", 0)

		if alg_cs < ref_cs:
			raise Exception("Error for Cart Pole AUC for SARSA: AUC_REF=", ref_cs, " ,AUC_ALG", alg_cs)

		if alg_cq < ref_cq:
			raise Exception("Error for Cart Pole AUC for QLearning: AUC_REF=", ref_cq, " ,AUC_ALG", alg_cq)
		#plt.show()

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print("Ctrl+C pressed; Exiting...")
	except Exception as e:
		print(str(e))