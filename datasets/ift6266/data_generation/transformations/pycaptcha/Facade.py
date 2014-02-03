#!/usr/bin/env python
import sys, os
curdir = os.path.dirname(__file__)
if curdir != '':
    sys.path.append(curdir)

from Captcha.Visual.Tests import PseudoGimpy, AngryGimpy
import numpy

# Une fonction simple pour generer un captcha
# ease : represente la difficulte du captcha a generer 
#      0 = facile et 1 (ou autre chose) = difficile 
#solution : specifie si on veut en retour un array numpy representant 
#l image ou un tuple contenant l'array et la solution du captcha.

# Des fontes additionnelles peuvent etre ajoutees au dossier pyCaptcha/Captcha/data/fonts/others
# Le programme choisit une fonte aleatoirement dans ce dossier ainsi que le dossir vera.


def generateCaptcha (ease=0, solution=0):

    if ease == 1:
        g = AngryGimpy()

    else:
        g = PseudoGimpy()
    
    i = g.render()
    a = numpy.asarray(i)

    if solution == 0:
       return a

    else :
        return (a, g.solutions)
