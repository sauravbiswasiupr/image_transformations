#!/usr/bin/python
# coding: utf-8

'''
Script qui calcule la proportion de chiffres, lettres minuscules et lettres majuscules
dans NIST train et NIST test.

Sylvain Pannetier Lebeuf dans le cadre de IFT6266, hiver 2010

'''

from pylearn.io import filetensor as ft
import matplotlib.pyplot as plt


#f1 = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/all_train_labels.ft')
f1 = open('/data/lisa/data/nist/by_class/all/all_train_labels.ft')
train = ft.read(f1)
#f2 = open('/home/sylvain/Dropbox/Msc/IFT6266/donnees/all_test_labels.ft')
f2 = open('/data/lisa/data/nist/by_class/all/all_test_labels.ft')
test = ft.read(f2)
f1.close()
f2.close()

#Les 6 variables
train_c=0
train_min=0
train_maj=0

test_c=0
test_min=0
test_maj=0

classe=0   #variable utilisee pour voir la classe presentement regardee
#Calcul pour le train_set
for i in xrange(len(train)):
    classe=train[i]
    if classe < 10:
        train_c += 1
    elif classe < 36:
        train_maj += 1
    elif classe < 62:
        train_min += 1

for j in xrange(len(test)):
    classe=test[j]
    if classe < 10:
        test_c += 1
    elif classe < 36:
        test_maj += 1
    elif classe < 62:
        test_min += 1
print "Train set:",len(train),"\nchiffres:",float(train_c)/len(train),"\tmajuscules:",\
float(train_maj)/len(train),"\tminuscules:",float(train_min)/len(train),\
"\nchiffres:", float(train_c)/len(train),"\tlettres:",float(train_maj+train_min)/len(train)

print "\nTest set:",len(test),"\nchiffres:",float(test_c)/len(test),"\tmajuscules:",\
float(test_maj)/len(test),"\tminuscules:",float(test_min)/len(test),\
"\nchiffres:", float(test_c)/len(test),"\tlettres:",float(test_maj+test_min)/len(test)


if test_maj+test_min+test_c != len(test):
    print "probleme avec le test, des donnees ne sont pas etiquetees"
    
if train_maj+train_min+train_c != len(train):
    print "probleme avec le train, des donnees ne sont pas etiquetees"


#train set
plt.subplot(211)
plt.hist(train,bins=62)
plt.axis([0, 62,0,40000])
plt.axvline(x=10, ymin=0, ymax=40000,linewidth=2, color='r')
plt.axvline(x=36, ymin=0, ymax=40000,linewidth=2, color='r')
plt.text(3,36000,'chiffres')
plt.text(18,36000,'majuscules')
plt.text(40,36000,'minuscules')
plt.title('Train set')

#test set
plt.subplot(212)
plt.hist(test,bins=62)
plt.axis([0, 62,0,7000])
plt.axvline(x=10, ymin=0, ymax=7000,linewidth=2, color='r')
plt.axvline(x=36, ymin=0, ymax=7000,linewidth=2, color='r')
plt.text(3,6400,'chiffres')
plt.text(18,6400,'majuscules')
plt.text(45,6400,'minuscules')
plt.title('Test set')

#afficher
plt.show()