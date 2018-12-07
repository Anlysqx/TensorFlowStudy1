import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(-1,1,100)
train_Y = train_X*3 + np.random.randn(*train_X.shape)*0.3
# plt.plot(train_X,train_Y,'ro',label='Original data')
# plt.legend()
# plt.show()
X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")

z = tf.multiply(X,W) + b
cost = tf.reduce_mean(tf.square(Y - z))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

init = tf.global_variables_initializer()
training_epochs = 100
display_step = 2

with tf.Session() as sess:
    sess.run(init)
    plotData = {'batchsize':[],
                'loss':[]}
    #向模型输入数据
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        #显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print('Epoch:',epoch+1,'cost=',loss,'W=',sess.run(W),'b=',sess.run(b))
            if not loss=='NA':
                plotData['batchsize'].append(epoch)
                plotData['loss'].append(loss)
    print('finished')
    print('cost=', sess.run(cost,feed_dict={X:train_X,Y:train_Y}), 'W=', sess.run(W), 'b=', sess.run(b))
    plt.plot(train_X,train_Y,'ro',label='Original_data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
    plt.show()
    plt.plot(plotData['batchsize'],plotData['loss'])
    plt.show()
    #使用模型
    print('hello world')
