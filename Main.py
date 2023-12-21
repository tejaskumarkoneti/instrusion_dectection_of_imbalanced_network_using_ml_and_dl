import keras
import tkinter as tk
from tkinter import Message ,Text
from tkinter import *
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.font as font
from tkinter import filedialog
import tkinter.messagebox as tm
from tkinter import ttk
import time
import matplotlib.pyplot as plt


import Preprocess as pre
import LRALG as lr
import RFALG as rf
import DTALG as dt
import KNNALG as knn
import NeuralNetwork as alexnet
import CNNLSTM as lstm
import adaboost as ada
import XGBoost as xgb


fontScale=1
fontColor=(0,0,0)
cond=0

bgcolor="#d7837f"
fgcolor="white"

window = tk.Tk()
window.title("Network Intrusion Detection")
#bg=PhotoImage(file="bg.png")


 
window.geometry('1280x720')
window.configure(background=bgcolor)
#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
#label1=tk.Label(window,image=bg)
#label1.place(x=0,y=0)


message1 = tk.Label(window, text="Network Intrusion Detection" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
message1.place(x=100, y=10)

lbl = tk.Label(window, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
lbl.place(x=10, y=200)

txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=300, y=215)




def browse():
	path=filedialog.askopenfilename()
	print(path)
	txt.delete(0, 'end')
	txt.insert('end',path)
	if path !="":
		print(path)
	else:
		tm.showinfo("Input error", "Select Dataset")	

	
def clear():
	txt.delete(0, 'end') 


def preprocess():
	sym=txt.get()
	if sym != "" :
		pre.process(sym)
		tm.showinfo("Input", "Preprocess Successfully Finished")
	else:
		tm.showinfo("Input error", "Select Dataset")



def rfprocess():
	rf.process()
	tm.showinfo("Input", "Random Forest Successfully Finished")
	
def dtprocess():
	dt.process()
	tm.showinfo("Input", "Decision Tree Successfully Finished")

def lrprocess():
	lr.process()
	tm.showinfo("Input", "Logistic Regression Successfully Finished")
		
def knnprocess():
	knn.process()
	tm.showinfo("Input", "KNN Successfully Finished")
def alexprocess():
	alexnet.process()
	tm.showinfo("Input", "Alexnet Successfully Finished")
def cnnlstmprocess():
	lstm.process()
	tm.showinfo("Input", "CNNLSTM Successfully Finished")
def adaprocess():
	ada.process()
	tm.showinfo("Input", "ADABOOST Successfully Finished")
def xgbprocess():
	xgb.process()
	tm.showinfo("Input", "XGBOOST Successfully Finished")
def comp():
        data=[76.50,75.646,62.74,73.224,99.16,99.70,57.58,76.31]
        labels=["Random Forest","Decision Tree","Logistic Regression","KNN","ALEXNET","CNN-LSTM","Adaboost","XGBoost"]
        plt.xticks(range(len(data)),labels)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
        plt.ylabel("Accuracy")
        plt.xlabel("Algorithms")
        plt.title("Comparision of Algorithm Accuracy")
        plt.bar(range(len(data)),data,color=colors)
        plt.show()
        
        



browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
browse.place(x=650, y=200)


pre1 = tk.Button(window, text="Preprocess", command=preprocess  ,fg=fgcolor  ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
pre1.place(x=10, y=500)

texta = tk.Button(window, text="Random Forest", command=rfprocess  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta.place(x=200, y=500)

texta1 = tk.Button(window, text="Decision Tree", command=dtprocess  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta1.place(x=400, y=500)


texta2 = tk.Button(window, text="Logistic Regression", command=lrprocess  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta2.place(x=600, y=500)

texta3 = tk.Button(window, text="KNN", command=knnprocess  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta3.place(x=820, y=500)
texta4 = tk.Button(window, text="Alex-Net", command=alexprocess  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta4.place(x=10, y=600)

texta5 = tk.Button(window, text="LSTM", command=cnnlstmprocess  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta5.place(x=200, y=600)


texta6 = tk.Button(window, text="AdaBoost", command=adaprocess  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta6.place(x=400, y=600)

texta7 = tk.Button(window, text="XGBoost", command=xgbprocess  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta7.place(x=620, y=600)
texta8 = tk.Button(window, text="Performance Evaluation", command=comp  ,fg=fgcolor ,bg=bgcolor  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
texta8.place(x=820, y=600)




quitWindow = tk.Button(window, text="QUIT", command=window.destroy  ,fg=fgcolor ,bg=bgcolor  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1050, y=600)

 
window.mainloop()
