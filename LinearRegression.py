#importing required dependencies
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#interactive plot
plt.ion()
ses=tf.Session()
x=np.array([int(i) for i in input("enter x values:").split(',')])
y=np.array([int(i) for i in input("enter y values:").split(',')])
#initializing tensor variables
X=tf.placeholder('float',[None])
Y=tf.placeholder('float',[None])
W=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.random_normal([1]))
#predicting the y
pred=tf.multiply(W,X)+b
#loss function
cost=tf.reduce_mean(tf.square(pred-y))
#gradient descent optimizer
optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#creating subplot 
fig,ax=plt.subplots(1,2)
init=tf.global_variables_initializer()
ses.run(init)
cost_=[]
iter_=[]
#updating the graph through each iteration 
def update(cost_,iter_):
    ax[0].cla()
    ax[1].cla()
    ax[0].set_title("Regression Line")
    ax[1].set_title("loss curve")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("loss")
    ax[0].set_ylim([0,max(y)+5])
    ax[0].scatter(x,y)
    ax[0].plot(x,y_)
    ax[1].set_xlim([0,200])
    ax[1].set_ylim([0,10])
    ax[1].plot(iter_,cost_)
    fig.canvas.draw()
#iteration steps here i takes 200 iterations
for i in range(0,200):
    y_,_,co=ses.run([pred,optimizer,cost],feed_dict={X:x,Y:y})
    cost_.append(co)
    iter_.append(i)
    update(cost_,iter_)
    
