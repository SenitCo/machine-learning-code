#encoding:utf-8

from numpy import *
from Tkinter import *
import regTrees

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():     # 复选框选中，则为模型树，否则会回归树
        if tolN < 2:
            tolN = 2
        myTree = regTrees.createTree(reDraw.rawData, regTrees.modelLeaf, regTrees.modelError, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testData, regTrees.modelTreeEval)
    else:
        myTree = regTrees.createTree(reDraw.rawData, ops = (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testData)
    reDraw.a.scatter(reDraw.rawData[:, 0], reDraw.rawData[:, 1], s=5)
    reDraw.a.plot(reDraw.testData, yHat, linewidth=2.0)
    reDraw.canvas.show()

def getInputs():
    try:
        tolN = int(tolNEntry.get())
    except:
        tolN = 10
        print "Enter Integer for tolN"
        tolNEntry.delete(0, END)
        tolNEntry.insert(0, '10')
    try:
        tolS = float(tolSEntry.get())
    except:
        tolS = 1.0
        print "Enter Float for tolS"
        tolSEntry.delete(0, END)
        tolSEntry.insert(0, '1.0')
    return tolN, tolS

def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

root = Tk()

reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)
tolNEntry = Entry(root)
tolNEntry.grid(row=1, column=1)
tolNEntry.insert(0, '10')
Label(root, text="tolS").grid(row=2, column=0)
tolSEntry = Entry(root)
tolSEntry.grid(row=2, column=1)
tolSEntry.insert(0, '1.0')
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawData = mat(regTrees.loadDataSet('sine.txt'))
reDraw.testData = arange(min(reDraw.rawData[:, 0]), max(reDraw.rawData[:, 0]), 0.01)
reDraw(1.0, 10)
root.mainloop()
