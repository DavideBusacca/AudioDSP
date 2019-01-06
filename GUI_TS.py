import tkinter as tk
from tkinter import ttk
import sys, os
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'TS'))

import utils as U
import phaseVocoder as PV
import overlap_add as OLA
  
#Declaring parameters name and initial values  
TS_methods = ["PhaseVocoder", "PhaseVocoderIdentity", "Overlap-Add"]
parametersStr = ['nameInput', 'nameOutput']
parametersStrInit = ['sounds/sine.wav', 'TS/processed/sine_stretched_PV.wav']
parametersInt = ['Hs', 'frameSize']
parametersIntInit = [256, 2048]
parametersDouble = ['alpha']
parametersDoubleInit = [1.5]

def printAll(comboMethod, myStrings, myInts, myDoubles):
    print(comboMethod.get() + " (" + str(comboMethod.current()) + ")")
    for string in myStrings:
        print(string.get())
    for numberInt in myInts:
        print(numberInt.get())
    for numberDouble in myDoubles:
        print(numberDouble.get())
           
        
def mainCallback(comboMethod, myStrings, myInts, myDoubles):
    #Print parameters
    method = comboMethod.get()
    print("Method is:\t\t'" + method + "'")
    
    nameInput = myStrings[0].get()
    nameOutput = myStrings[1].get()
    print("NameInput is:\t\t'" + nameInput + "'")
    print("NameOutput is:\t\t'" + nameOutput + "'")      
    
    Hs = myInts[0].get()
    frameSize = myInts[1].get()
    print("HopeSizeS is:\t\t" + str(Hs))
    print("FrameSize is:\t\t" + str(frameSize))
        
    alpha = myDoubles[0].get()   
    print("Alpha is:\t\t" + str(alpha))
    
    start_time = time.time()
    #Calling to the Time-Stretching Functions
    if method == "PhaseVocoder": 
        PV.callback(nameInput=nameInput, nameOutput=nameOutput, alpha=alpha, Hs=Hs, frameSize=frameSize, phaseLocking='no')
    elif method == "PhaseVocoderIdentity":
        PV.callback(nameInput=nameInput, nameOutput=nameOutput, alpha=alpha, Hs=Hs, frameSize=frameSize, phaseLocking='identity')
    elif method == "Overlap-Add":
        OLA.callback(nameInput=nameInput, nameOutput=nameOutput, alpha=alpha, Hs=Hs, frameSize=frameSize)
    else:
        print("Congratulation! I don't know how you get here, but the Time-Stretching function you are asking me is not included or does not exist!")
    end_time = time.time()    

    print("Time elapsed for computation: " + str(end_time-start_time) + " seconds (visualization computation included) \n")

if __name__ == "__main__":
    #Create Window object
    window=tk.Tk()

    myStrings=[]
    myInts=[]
    myDoubles=[]
    labels=[]
    entries=[]
    
    comboMethod = ttk.Combobox(window, 
                            values=TS_methods)

    comboMethod.grid(column=0, row=0)
    comboMethod.current(0) #set the 0st value as chosen
    
    #Define labels
    for idx, nameParameter in enumerate(parametersStr):
        l=tk.Label(window, text=nameParameter)
        l.grid(row=idx, column=1)
        inputString=tk.StringVar()
        e=tk.Entry(window, textvariable=inputString)
        e.grid(row=idx, column=2)
        e.delete(0, tk.END)
        e.insert(0, parametersStrInit[idx])
        labels.append(l)
        entries.append(e)
        myStrings.append(inputString)
    
    for idx, nameParameter in enumerate(parametersInt):
        l=tk.Label(window, text=nameParameter)
        l.grid(row=idx, column=3)
        inputInt=tk.IntVar()
        e=tk.Entry(window, textvariable=inputInt)
        e.grid(row=idx, column=4)
        e.delete(0, tk.END)
        e.insert(0, parametersIntInit[idx])
        labels.append(l)
        entries.append(e)        
        myInts.append(inputInt)
        
    for idx, nameParameter in enumerate(parametersDouble):
        l=tk.Label(window, text=nameParameter)
        l.grid(row=idx, column=5)
        inputDouble=tk.DoubleVar()
        e=tk.Entry(window, textvariable=inputDouble)
        e.grid(row=idx, column=6)
        e.delete(0, tk.END)
        e.insert(0, parametersDoubleInit[idx])
        labels.append(l)
        entries.append(e)
        myDoubles.append(inputDouble)
    
    #Define buttons
    b1=tk.Button(window, text='Run!', width=12, command=lambda: mainCallback(comboMethod, myStrings, myInts, myDoubles))
    b1.grid(row=0, column=7)    
    
    b2=tk.Button(window, text='Print All Parameters', width=12, command=lambda: printAll(comboMethod, myStrings, myInts, myDoubles))
    b2.grid(row=1, column=7)    
        
    b3=tk.Button(window, text='Play Input', width=12, command=lambda: U.wavplay(myStrings[0].get()))
    b3.grid(row=2, column=7)
    
    b4=tk.Button(window, text='Play Output', width=12, command=lambda: U.wavplay(myStrings[1].get()))
    b4.grid(row=3, column=7)    
    
    #Run
    window.mainloop()
