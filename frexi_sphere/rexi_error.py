# Based on Colins code
#
# 2017-03-12: MaS: Some modifications to get error


from rexi_coefficients import *
import cmath
import numpy as np

h = 0.2
M = 64

# load coefficients *WITHOUT* reduction to the half
alpha, beta_re, beta_im = RexiCoefficients(h, M, False)

xmax = 1.5
xmax = 0.001
#ymax = 150.0
ymax = 15.0

#res=100
res=100

xs = np.arange(-xmax,xmax,xmax/res)
ys = np.arange(-ymax,ymax,ymax/res)

evalues = np.zeros((len(xs), len(ys)), dtype="complex")
for i in range(0, len(xs)):
	for j in range(0, len(ys)):
		evalues[i,j] = 1j*ys[i]+xs[j]


stability = np.zeros((len(xs), len(ys)))
error_re_including = np.zeros((len(xs), len(ys)))
error_re_excluding = np.zeros((len(xs), len(ys)))

for i in range(0, len(xs)):
	for j in range(0, len(ys)):
            evalue = evalues[i,j]
	    sumval = 0.

	    # compute REXI SUM
	    for n in range(len(alpha)):
		denom = (evalue + alpha[n]);
		sumval += (beta_re[n] / denom).real + 1j*(beta_im[n] / denom).real
		#sumval += (beta_re[n] / denom).real

	    # Check for stability and add if stable
	    if abs(sumval) <= 1.0:
            	stability[i,j] = 1

	    # analytical solution
	    val = np.exp(evalue)
	    error_re_including[i,j] = abs((val-sumval).real)

	    val2 = np.exp(evalue.imag*1j)
	    error_re_excluding[i,j] = abs((val2-sumval).real)




def heatmap_plot(
		title,
		xlabel,
		col_labels,
		ylabel,
		row_labels,
		data,
		outfile = '',
		legend_min = None,
		legend_max = None
		):
        import matplotlib.pyplot as plt
	from matplotlib import cm
	from matplotlib.colors import LogNorm

	col_labels = [str(i) for i in col_labels]
	mod = len(col_labels)/10
	for i in range(0, len(col_labels)):
		if i % mod > 0:
			col_labels[i] = ''
			continue
		if abs(float(col_labels[i])) < 1e-10:
			col_labels[i] = 0
			
	row_labels = [str(i) for i in row_labels]
	mod = len(row_labels)/10
	for i in range(0, len(row_labels)):
		if i % mod > 0:
			row_labels[i] = ''
			continue
		if abs(float(row_labels[i])) < 1e-10:
			row_labels[i] = 0

	fig, ax = plt.subplots()

	
	legend_norm = LogNorm(legend_min, legend_max)
	heatmap = ax.pcolor(data, cmap=cm.rainbow, norm=legend_norm)

	plt.title(title, y=1.08, fontweight='bold')

	#legend
	cbar = plt.colorbar(heatmap)
#	cbar.set_label('in height', rotation=270)

	fig.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.3)

	# put the major ticks at the middle of each cell
	ax.set_xticks(np.arange(len(col_labels))+0.5, minor=False)
	ax.set_xticklabels(col_labels, minor=False, rotation=45)
	ax.set_xlim(0, len(col_labels))
	ax.set_xlabel(xlabel)

	ax.set_yticks(np.arange(len(row_labels))+0.5, minor=False)
	ax.set_yticklabels(row_labels, minor=False)
	ax.set_ylim(0, len(row_labels))
	ax.set_ylabel(ylabel)

	# want a more natural, table-like display
	ax.invert_yaxis()


	if outfile != '':
		plt.savefig(outfile, dpi=300)#, format='pdf')
	else:
		plt.show()


heatmap_plot("REXI stability", "Re(evalue)", xs, "Im(evalue)", ys, stability, "output_rexi_stability.png", 1e-10, 1)

heatmap_plot("REXI error - |Re(REXI(evalue)-exp(evalue))|", "Re(evalue)", xs, "Im(evalue)", ys, error_re_including, "output_rexi_error_re_including.png", 1e-10, 1)

heatmap_plot("REXI error - |Re(REXI(evalue)-exp(evalue.imag))|", "Re(evalue)", xs, "Im(evalue)", ys, error_re_excluding, "output_rexi_error_re_excluding.png", 1e-10, 1)

#heatmap_plot("REXI error (imag)", "Im(evalue)", xs, "Re(evalue)", ys, error_im, "output_rexi_error_im.png", 1e-10, 1)


