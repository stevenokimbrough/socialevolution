#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:41:56 2019

@author: kimbrough

File name: voweltraditionsobj.py
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
def drawAVowel(name='/c/', n=1, mu = [700,900], 
               sigma = [[1700,10000],[10000,26000]] ):
    '''
    Draws a sample from a multivariate Gaussian with mean mu and covariance 
    matrix, sigma. n is the sample size.
    '''
    daVowel=np.random.multivariate_normal(mu, sigma, n)
    return (name,daVowel)

def updateDistribution(population):
    '''
    population is an arrray of two columns. The first column is
    the F1 of an utterance, the second is the F2.
    
    We want to estimate the mu and sigma values for the population.
    
    The purpose here is to update the representation of a vowel, which
    is the mean and covariance, which are returned by this procedure.
    '''
    sigma = np.cov(population[:,0],population[:,1])
    mu = population.mean(axis=0)
    return (mu,sigma)

class Vowel():
    '''
    
    '''
    
    def __init__(self,aname='/i/', amu= [400, 2700], 
                 asigma = [[800,3000],[3000,56000]]):
        self.name = aname
        self.mu = amu
        self.sigma = np.array(asigma)
        
        
    def drawAVowel(self, name, mu, 
                   sigma, size = 1):
        '''
        Draws a sample from a multivariate Gaussian with mean mu and covariance 
        matrix, sigma. n is the sample size.
        '''
        daVowelInstance=np.random.multivariate_normal(mu, sigma, size)
        return daVowelInstance

        
        
        
class Conversant():
    '''
    Class for representing speakers and hears, conversants.
    '''
    
    def __init__(self,vowelSystem,ID=None):
        '''
        vowelSystem should be a list of tuples:
            (vowel name, mu, sigma). We process the list
            at initialization to get vowelSpace, a dictionary.
            Each vowel in vowelSpace is represented by a triple:
                (vowel name, mu, sigma)
        '''
        
        
        self.ID = ID
        self.vowelSystem = {}
        self.vowelHistory = {} # key is vname, value (mu,sigma)
        
        '''
        This is the current vowel system, which is updated as we go.
        For each vowel we have key: vowel name, value is a 2-d array
        of formant values. This gets changed as the agent listens. 
        
        So vowelSystem is a dictionary. key: vowel name, value is a tuple:
        (mu array, sigma array).
        
        vowelSystem is a dictionary holding mu and sigma for individual 
        vowels.
        
        vowelPopulations is a dictionary, key is vowel name, value is
        a 2-d (with 2 formants) array of frequencies. For now, 50 long,
        soon: abstracted.
        
        vowelHistory is a list of vowelSystem value.
        '''

        
        # Initial recording of the (given) vowel mu and sigma
        for item in vowelSystem:
            self.vowelSystem[item[0]] = (item[1],item[2])
            #Our new pattern is (round number, mu, sigma)
# =============================================================================
#             self.mu = item[1]
#             self.sigma = item[2]
# =============================================================================
        self.initialVowelSystem = self.vowelSystem.copy()
        
        # Now, generate for each given vowel a sample of utterances.  
        self.vowelPopulations = {}
        for vname in self.vowelSystem:
            vowelObj = Vowel()
            (mu,sigma) = self.vowelSystem[vname]
            vpop = vowelObj.drawAVowel(vname, mu, sigma, size = 50)
            self.vowelPopulations[vname] = vpop
            (newmu,newsigma) = updateDistribution(vpop)
            self.vowelSystem[vname] = (newmu,newsigma)
            currentHistory = self.vowelHistory.get(vname,[])
            currentHistory.append((0,newmu,newsigma))
            self.vowelHistory[vname] = currentHistory
            
    def getMuSigma(self,vowelName):
        (mu,sigma) = self.vowelSystem[vowelName]
        return (mu,sigma)
        
    def guessVowel(self):
        '''
        Let's not forget to fix this.
        '''
        return '/i/'            
            
    def getMuHistory(self,vowelName):
        '''
        Given a vowel name, returns a list of its values from the
        vowelHistory
        '''
        histList = self.vowelHistory[vowelName]
        #histList = hist[vowelName] # histList is a list of 2-tuples, formants
        toReturn = []
        for item in histList:
            formantValues = list(item[1])
            toReturn.append([item[0]]+formantValues)
            
        return np.array(toReturn)
        
        
        
        
        # Estimate the mu and sigma values from the sample utterances.
            
            
class VowelLibrary():
    '''
    Underlying library of default vowels of various sorts. 
    To be expanded as needed.
    '''
    
    def __init__(self):
        self.vowelDict = {}
        self.vowelDict['/c/'] = ([400, 2700], [[800,3000],[3000,56000]])
        self.vowelDict['/i/'] = ([400, 2700], [[900,3300],[3300,58000]])
        
    def getVowel(self,name):
        (mu,sigma) = self.vowelDict[name] 
        return (name,mu,sigma)
    
    def getVowelList(self):
        return list(self.vowelDict.keys())
        
            
            
def getAVowel(name,dictionary):
    (mu,sigma)=dictionary[name]
    return (name,mu,sigma)
    
def speak(guy):
    '''
    The guy picks a vowel from its vowelSystem 
    and draws an utterance.
    '''
    
    # for now:
    vNameList = np.array(list(guy.vowelPopulations.keys()))
    np.random.shuffle(vNameList)
    vName = vNameList[0]
    # but for now:
    vName = '/i/'
    (mu,sigma) = guy.getMuSigma(vName)
    (name,vowelInstance) = drawAVowel(name=vName, n=1, mu = mu, 
               sigma = sigma )   
    return (name,vowelInstance)

def perceive(guy,utterance,daRound):
    '''
    guy is an agent (Conversant) and utterance a (name,vowelInstance)
    '''
    vGuessed = guy.guessVowel() # Right now, this is only '/i/'
    daPop = guy.vowelPopulations[vGuessed]
    (rows,_) = daPop.shape
    index = np.random.randint(rows)
    # So now replace row index
    daPop[index,:] = utterance[1]
    guy.vowelPopulations[vGuessed] = daPop
    
    # Now update mu and sigma
    newsigma = np.cov(daPop[:,0],daPop[:,1])
    newmu = daPop.mean(axis=0)
    #print("New mu and Sigma",(newmu,newsigma))


    currentHistory = guy.vowelHistory.get(vGuessed,[])
    currentHistory.append((daRound,newmu,newsigma))
    guy.vowelHistory[vGuessed] = currentHistory    
    # Update the guy's vowelSystem
    guy.vowelSystem[vGuessed] = (newmu,newsigma)

def doit(interactions = 1):
    '''
    Drives the action.
    '''
# =============================================================================
#     # Get an initial vowel
#     vl = VowelLibrary()
#     (name,mu,sigma) = vl.getVowel('/i/')
# =============================================================================
    
    # Grab the vowel library:
    vowelList = []
    vl = VowelLibrary()
    for name in vl.vowelDict:
        (mu, sigma) = vl.vowelDict[name]
        vowelList.append((name,mu,sigma))
        
    # Initialize the two Conversants.
    # Make a dictionary of Conversants.
    # The keys are the conversant IDs, which begin with 1.
    # The values are the conversant objects. Here is
    # also where we create these object insances.
    Conversants = {}
    for c in range(2):
        Conversants[c+1] = Conversant(vowelList,c+1)

    # print out mu and sigma for each conversant
    # seems good:
    print("Initial mu and sigma:")
    for ID in Conversants.keys():
        guy = Conversants[ID]
        for vname in guy.vowelSystem.keys():
            print(ID,vname, guy.getMuSigma(vname))
    print("*******************")
# =============================================================================
#     utterance = speak(Conversants[1])
#     perceive(Conversants[2],utterance)
# =============================================================================

# Guys talking to themselves:
# =============================================================================
#     for rnd in range(1,6000):
#         IDs = list(Conversants.keys())       
#         sindex = np.random.randint(len(IDs))
#         speakerKey = IDs[sindex]
#         speaker = Conversants[speakerKey]
#         hearer = speaker
#         utterance = speak(speaker)
#         perceive(hearer,utterance,rnd)
#         
#         IDs.remove(speakerKey)
#         hindex = np.random.randint(len(IDs))
#         hearerKey  = IDs[hindex]
#         hearer = Conversants[hearerKey]
#         speaker = hearer
#         #print(rnd,'speaker,hearer',speakerKey,hearerKey)
#         utterance = speak(speaker)
#         perceive(hearer,utterance,rnd)
# =============================================================================
        
# Guys talking to each other:
    for rnd in range(1,60000):
        IDs = list(Conversants.keys())
    
        sindex = np.random.randint(len(IDs))
        speakerKey = IDs[sindex]
        speaker = Conversants[speakerKey]
        IDs.remove(speakerKey)
        hindex = np.random.randint(len(IDs))
        hearerKey  = IDs[hindex]
        hearer = Conversants[hearerKey]
        #print(rnd,'speaker,hearer',speakerKey,hearerKey)
        utterance = speak(speaker)
        perceive(hearer,utterance,rnd)
        
        
        
    return Conversants

#%%
def plotMu0(guys):
    '''
    guys, a dict of conversants
    '''
    plt.figure()
    guyList = list(guys.keys())
    guy0 = guys[guyList[0]]
    x = guy0.getMuHistory('/i/')[:,0].astype(int)
    y = guy0.getMuHistory('/i/')[:,1]
    plt.plot(x,y)

    guy0 = guys[guyList[1]]
    x = guy0.getMuHistory('/i/')[:,0].astype(int)
    y = guy0.getMuHistory('/i/')[:,1]
    plt.plot(x,y)    
    plt.show()
#%%
if __name__ == '__main__':
#    bob = Conversant([(1,2),(3,4)],123)     
#    print(bob.vowelSystem)    
    daGuys = doit()
#    bob = daGuys[1]
 #   carol = daGuys[2]
# =============================================================================
#     print(bob.vowelHistory)
#     print(carol.vowelHistory)
# =============================================================================

    plotMu0(daGuys)
 
#    bob = Vowel()
#    
#    print(bob.name)
#    carol = VowelLibrary()
#    print(carol.getVowelList())
#    (name,mu,sigma) = (carol.getVowel('/i/'))
        
        
    