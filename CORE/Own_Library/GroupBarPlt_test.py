import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
mpl.rcParams['figure.figsize'] = [10, 7]

def groupBarPlot(plotname, algoname, filename = None, df = None):
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Plot/')
	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)

	if filename == None:
		df2 = df
	else:
		df2 = pd.read_csv(filename)

	df2 = df2.round(4)
	ax = df2.plot(kind ='bar', x=df2.columns[0], width = 0.9) # y=df2.columns[1:],
	for container in ax.containers:
		ax.bar_label(container)
		
	plt.title(f"{plotname} Plot for {algoname}")
	plt.xlabel("\n"+df2.columns[0])
	plt.ylabel(plotname)
	plt.xticks(rotation = 0)
	# plt.ylim(0,1.3)
	plt.legend(loc ='best')
	# plt.show()
	plt.savefig(results_dir+f'{plotname}_{algoname}.jpg')