#!/usr/bin/python

import sys
import fnmatch
import os
import glob
import shutil
import argparse
import time
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pylab as pl
from matplotlib.ticker import LogLocator, MultipleLocator, FormatStrFormatter

currentdir = os.getcwd()

parser = argparse.ArgumentParser(description='Visualizes already\
computed benchmarks from pLA.')
parser.add_argument('-f', '--filename', required=True,
    help='Start of file names.')

args = parser.parse_args()

# go to directory
os.chdir(currentdir)

files = list()
# find bench file
for f in glob.glob(args.filename+'*'):
  files.append(f)

files.sort()
# read lines of the benchmark files
lines = list()
for i in range(0,len(files)):
  f = open(files[i]+'/bench.res')
  lines.append(f.readlines())
  f.close()
# get
# 1. dimensions of benchmark matrices
# 2. threads for plot, stored in the first line of bench file
dimensions = lines[0][0].strip().replace(' ','').split(',')
# second line are the thread settings used
plot_threads = lines[0][1].strip().replace(' ','').split(',')
# get algorithm benchmarked
algorithm_benched = lines[0][2].strip()
print algorithm_benched
# get threads for plot, stored in the first line of bench file
#plot_threads = f.readline().strip().replace(' ','').split(',')
# for compatibility to the other scripts just store this again
threads = plot_threads
plot_data = list()
start_threads = int(threads[0])
threads = list(map(lambda x: int(x) - 1, threads))

# list of all methods, sequential only if start_threads == 1
if start_threads == 1:
  methods = ['Sequential', 'Intel TBB','StarPU']
else :
  methods = ['Intel TBB', 'StarPU']
# lists for all methods we have, those are lists of lists:
# E.g. time_series[i] is a list of len(threads) elements of the timings
# of methods[i].

# first coordinate = blocksize
# second coordinate = methods
time_series = [[list() for x in xrange(len(methods))] for y in xrange(len(files))]
gflops_series = [[list() for x in xrange(len(methods))] for y in xrange(len(files))]
speedup_series = [[list() for x in xrange(len(methods))] for y in xrange(len(files))]
labels = list()
blocksize_series = list()

# test for incremental computations
tt = files[0].split('-')
if tt[3] == 'inc':
  inc = 1
else:
  inc = 0
for i in range(0,len(files)):
  bs = files[i].split('-')
  if inc == 0:
    blocksize_series.append(bs[5])
  else:
    blocksize_series.append(bs[6])


# labels for plots
for i in range(0,len(files)):
  for j in range(0,len(methods)):
    labels.append(methods[j]+' - bs '+blocksize_series[i])

gflops = \
    (float(2)*(float(dimensions[0])**3))/(float(3)*float(1000)*float(1000)*float(1000))
tmp = -1
tmpold = -1
for j in range(0,len(files)):
  for l in lines[j]:
    for i in range(0,len(methods)):  
      if l.find(methods[i]) != -1:
       tmpold = tmp
       tmp = i
    if l.find('Real time:') != -1:
      if tmpold < tmp:
        time_seq = l.replace('Real time:','').replace('sec','').strip()
        tmpold = tmp
      time_series[j][tmp].append(\
        l.replace('Real time:','').replace('sec','').strip())
    if l.find('GFLOPS/sec:') != -1:
      # if the value is inf for infinity due to short computation time, we set
      # the GFLOPS value to be -1
      gflops_series[j][tmp].append(\
        gflops/float(time_series[j][tmp][len(time_series[j][tmp])-1]))

if inc == 0:
  timings_seq = list()
  for j in range(0,len(files)):
    for i in range(0,len(methods)):
      timings_seq.append(float(time_series[j][i][0]))
  # get minimal sequential timing
  min_seq_time = min(timings_seq)
  for j in range(0,len(files)):
    for i in range(0,len(methods)):
      for k in range(0,len(time_series[j][i])):
        speedup_tmp =float(min_seq_time) / \
        float(time_series[j][i][k])
        speedup_series[j][i].append(str(speedup_tmp))

print(speedup_series[0])
#plot this data
#line style, sequential method only if start_threads == 1
stride = 1
coloring =\
[\
'#0099cc','#33cc00','#ff1b54','#0033cc','#9900cc','#800020',\
'#ff4011','#ffbf01','#00144f','#ff1450',\
'#0099cc','#33cc00','#cc0033','#0033cc','#9900cc','#800020',\
'#ff4011','#ffbf01','#00144f','#ff1450',\
'#0099cc','#33cc00','#cc0033','#0033cc','#9900cc','#800020',\
'#ff4011','#ffbf01','#00144f','#ff1450',\
'#0099cc','#33cc00','#cc0033','#0033cc','#9900cc','#800020',\
'#ff4011','#ffbf01','#00144f','#ff1450'\
]
print(threads)
if threads[0]+1 == 1:
  styles = ['-','-','--',':']
  markers = [\
  'o','None','None','None','None','None','None','None','None','None',\
  'None','None','None','None','None','None','None','None','None','None',\
  'None','None','None','None','None','None','None','None','None','None',\
  'o','o','o','o','o','o','o','o','o','o'\
  ]
else:
  styles = ['-','--',':']
  markers = [\
  'None','None','None','None','None','None','None','None','None',\
  'None','None','None','None','None','None','None','None','None','None',\
  'None','None','None','None','None','None','None','None','None','None',\
  'o','o','o','o','o','o','o','o','o','o'\
  ]
  


pl.rc('legend',**{'fontsize':5})
fig = pl.figure()
ax = fig.add_subplot(111)
fig.suptitle('Timings: '+args.filename, fontsize=10)
if inc == 0:
  pl.title('Tiled GEP uint64 Matrix dimensions: '+dimensions[0]+
   ' x '+dimensions[1], fontsize=8)
else:
  pl.title('Tiled GEP uint64 Matrix dimensions: '+dimensions[0]+
  ' x '+dimensions[1]+' increasing by '+dimensions[2]+' in each step using '+
  str(max_threads)+' threads', fontsize=8)
if inc == 0:
  ax.set_xlabel('Number of threads', fontsize=7)
else:
  ax.set_xlabel('Matrix sizes', fontsize=7)
ax.set_ylabel('Real time in seconds', fontsize=8)

pl.grid(b=True, which='major', color='k', linewidth=0.3)
pl.grid(b=True, which='minor', color='k', linewidth=0.1, alpha=0.5)

ax = pl.gca() 

group_labels = plot_threads

#ax.set_xticklabels(group_labels)
threads_tmp = range(0,len(plot_threads))
# get right scale for a4 paper size
scale_tmp = 38 / (len(plot_threads)) 
threads = range(0,38,scale_tmp)
tick_lbs = plot_threads
ax.xaxis.set_ticks(threads)
ax.xaxis.set_ticklabels(tick_lbs)
print(time_series[0])
print(threads)
p = [[[None] for x in xrange(len(methods))] for y in xrange(len(files))]
for j in range(0,len(files)):
  for i in range(0,len(methods)):
    p[j][i], = ax.semilogy(threads[0:len(time_series[j][i])], time_series[j][i],
            c=coloring[j],
      ls=styles[i], marker=markers[i], markersize='6', markevery=stride,
      label=i+j*len(methods), basey=2)
# set 0 as min value for y and 1 as min value for x (threads)
#pl.xlim(xmin=1)
ax.legend((labels),'upper right', shadow=True, fancybox=True)

# take real time of sequential computation to figure out the 
# granularity of the yaxis
pl.tick_params(axis='both', which='major', labelsize=8)
pl.tick_params(axis='both', which='minor', labelsize=8)

pl.savefig('timings-plot.pdf',papertype='a4',orientation='landscape')

fig = pl.figure()
ax = fig.add_subplot(111)
fig.suptitle('GFLOPS/sec: '+args.filename, fontsize=10)
if inc == 0:
  pl.title('Tiled GEP uint64 Matrix dimensions: '+dimensions[0]+
   ' x '+dimensions[1], fontsize=8)
else:
  pl.title('Tiled GEP uint64 Matrix dimensions: '+dimensions[0]+
  ' x '+dimensions[1]+' increasing by '+dimensions[2]+' in each step using '+
  str(max_threads)+' threads', fontsize=8)
if inc == 0:
  ax.set_xlabel('Number of threads', fontsize=7)
else:
  ax.set_xlabel('Matrix sizes', fontsize=7)
ax.set_ylabel('GFLOPS per second', fontsize=8)

pl.grid(b=True, which='major', color='k', linewidth=0.3)
pl.grid(b=True, which='minor', color='k', linewidth=0.1, alpha=0.5)

ax = pl.gca() 

#ax.set_xticklabels(group_labels)
threads_tmp = range(0,len(plot_threads))
# get right scale for a4 paper size
scale_tmp = 38 / (len(plot_threads)) 
threads = range(0,38,scale_tmp)
tick_lbs = plot_threads
ax.xaxis.set_ticks(threads)
ax.xaxis.set_ticklabels(tick_lbs)

p = [[[None] for x in xrange(len(methods))] for y in xrange(len(files))]
for j in range(0,len(files)):
  for i in range(0,len(methods)):
    p[j][i], = ax.semilogy(threads[0:len(gflops_series[j][i])], gflops_series[j][i],
            c=coloring[j],
      ls=styles[i], marker=markers[i], markersize='6', markevery=stride,
      label=i+j*len(methods), basey=2)
# set 0 as min value for y and 1 as min value for x (threads)
#pl.xlim(xmin=1)
pl.ylim(ymin=0)
ax.legend((labels),'upper left', shadow=True, fancybox=True)
# take real time of sequential computation to figure out the 
# granularity of the yaxis
pl.tick_params(axis='both', which='major', labelsize=8)
pl.tick_params(axis='both', which='minor', labelsize=8)


pl.savefig('gflops-plot.pdf',papertype='a4',orientation='landscape')

fig = pl.figure()
ax = fig.add_subplot(111)
fig.suptitle('Speedup: '+args.filename, fontsize=10)
if inc == 0:
  pl.title('Tiled GEP uint64 Matrix dimensions: '+dimensions[0]+
   ' x '+dimensions[1], fontsize=8)
else:
  pl.title('Tiled GEP uint64 Matrix dimensions: '+dimensions[0]+
  ' x '+dimensions[1]+' increasing by '+dimensions[2]+' in each step using '+
  str(max_threads)+' threads', fontsize=8)
if inc == 0:
  ax.set_xlabel('Number of threads', fontsize=7)
else:
  ax.set_xlabel('Matrix sizes', fontsize=7)
ax.set_ylabel('Speedup', fontsize=8)

pl.grid(b=True, which='major', color='k', linewidth=0.3)
pl.grid(b=True, which='minor', color='k', linewidth=0.1, alpha=0.5)

ax = pl.gca() 

#ax.set_xticklabels(group_labels)
threads_tmp = range(0,len(plot_threads))
# get right scale for a4 paper size
scale_tmp = 38 / (len(plot_threads)) 
threads = range(0,38,scale_tmp)
tick_lbs = plot_threads
ax.xaxis.set_ticks(threads)
ax.xaxis.set_ticklabels(tick_lbs)

p = [[[None] for x in xrange(len(methods))] for y in xrange(len(files))]
for j in range(0,len(files)):
  for i in range(0,len(methods)):
    p[j][i], = ax.semilogy(threads[0:len(speedup_series[j][i])],
            speedup_series[j][i], c=coloring[j],
      ls=styles[i], marker=markers[i], markersize='6', markevery=stride,
      label=i+j*len(methods), basey=2)
# set 0 as min value for y and 1 as min value for x (threads)
#pl.xlim(xmin=1)
pl.ylim(bottom=-2)
ax.legend((labels),'upper left', shadow=True, fancybox=True)

pl.tick_params(axis='both', which='major', labelsize=8)
pl.tick_params(axis='both', which='minor', labelsize=8)

pl.savefig('speedup-plot.pdf',papertype='a4',orientation='landscape')
