#!/usr/bin/env python

import os
dir1 = "/data/lisa8/data/ift6266h10/"
dir2="/data/lisa/data/ift6266h10/"

mach = "maggie16.iro.umontreal.ca,zappa8.iro.umontreal.ca,maggie15.iro.umontreal.ca,brams04.iro.umontreal.ca"

#test and valid sets
for i,s in enumerate(['valid','test']):
    for j,c in enumerate([0.3,0.5,0.7,1]):
        l = str(c).replace('.','')
        os.system("dbidispatch --condor --os=fc4,fc7,fc9 --machine=%s ./run_pipeline.sh -o %sdata/P%s_%s_data.ft -p %sdata/P%s_%s_params -x %sdata/P%s_%s_labels.ft -f %s%s_data.ft -l %s%s_labels.ft -c %socr_%s_data.ft -d %socr_%s_labels.ft -m %s -z 0.1 -a 0.1 -b 0.25 -g 0.25 -s %d -y %d" % (mach, dir1, l, s, dir1, l, s, dir1, l, s, dir1, s, dir1, s, dir1, s, dir1, s, c ,[20000,80000][i], 200+i*4+j))

#P07
for i in range(100):
    os.system("dbidispatch --condor --os=fc4,fc7,fc9 --machine=%s ./run_pipeline.sh -o %sdata/P07_train%d_data.ft -p %sdata/P07_train%d_params -x %sdata/P07_train%d_labels.ft -f %strain_data.ft -l %strain_labels.ft -c %socr_train_data.ft -d %socr_train_labels.ft -m 0.7 -z 0.1 -a 0.1 -b 0.25 -g 0.25 -s 819200 -y %d" % (mach, dir1, i, dir1, i, dir1, i, dir1, dir1, dir1, dir1, 100+i))

#PNIST07
for i in xrange(10,100):
   os.system("dbidispatch --condor --mem=3900 --os=fc4,fc7,fc9 --machine=%s ./run_pipeline.sh -o %sdata/PNIST07_train%d_data.ft -p %sdata/PNIST07_train%d_params -x %sdata/PNIST07_train%d_labels.ft -f %strain_data.ft -l %strain_labels.ft -c %socr_train_data.ft -d %socr_train_labels.ft -m 0.7 -z 0.1 -a 0.1 -b 0.25 -g 0.25 -s 819200 -y %d -t 1" % (mach, dir1, i, dir1, i, dir1, i, dir2, dir2, dir2, dir2, 100+i))



#PNIST_full_noise
for i in range(100):
   os.system("dbidispatch --condor --mem=3900 --os=fc4,fc7,fc9 --machine=%s ./run_pipeline.sh -o %sdata/Pin07_train%d_data.ft -p %sdata/Pin07_train%d_params -x %sdata/Pin07_train%d_labels.ft -f %strain_data.ft -l %strain_labels.ft -c %socr_train_data.ft -d %socr_train_labels.ft -m 0.7 -z 0.1 -a 0.1 -b 0.25 -g 0.25 -s 819200 -y %d" % (mach, dir1, i, dir1, i, dir1, i, dir2, dir2, dir2, dir2,100+i))



#P07_safe
for i in xrange(89,100,1):
   os.system("dbidispatch --condor --mem=3900 --os=fc4,fc7,fc9 --machine=%s ./run_pipeline.sh -o %sdata/P07safe_train%d_data.ft -p %sdata/P07safe_train%d_params -x %sdata/P07safe_train%d_labels.ft -f %strain_data.ft -l %strain_labels.ft -c %socr_train_data.ft -d %socr_train_labels.ft -m 0.7 -z 0.1 -a 0.0 -b 0.0 -g 0.0 -s 819200 -y %d" % (mach, dir1, i, dir1, i, dir1, i, dir2, dir2, dir2,dir2,100+i))

