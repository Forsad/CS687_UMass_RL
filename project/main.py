import os


def read_episode(f):
	episode = []
	epLen = int(f.readline())
	for st in range(epLen):
		ln = f.readline()[:-1]
		vals = ln.split(",")
		vals = (int(vals[0]), int(vals[1]), float(vals[2]), float(vals[3]))
		print(vals)
		episode.append(vals)
def read_data(dat_fl):
	ret = []
	with open(dat_fl, "r") as f:
		eps = int(f.readline())
		for ep in range(eps):
			episode = read_episode(f)
			ret.append(episode)
			break
	return ret


def main():
	dat_fl = "data.csv"
	dat = read_data(dat_fl)


if __name__ == '__main__':
	main()