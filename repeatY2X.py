import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_X = np.linspace(-1,1,100)
train_Y = train_X * 2 + np.random.randn(*train_X.shape)*0.3

inputdict = {
    'X' : tf.placeholder("float"),
    'Y' : tf.placeholder("float")
}
W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")

z = tf.multiply(inputdict['X'],W) + b
cost = tf.reduce_mean(np.square(inputdict['Y'] - z))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

init = tf.global_variables_initializer()
train_epoch = 50
plot_step = 2

with tf.Session() as sess:
    sess.run(init)
    plot_data = {'batch_num':[],
                 'loss':[]}
    for epoch in range(train_epoch):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        if epoch % plot_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print('epoch : ', epoch, " loss : ", loss, " w : ", sess.run(W), " b: ", sess.run(b))
            plot_data['batch_num'].append(epoch)
            plot_data['loss'].append(loss)
    print('train finished')
    print(" loss : ",sess.run(cost,feed_dict={X:train_X,Y:train_Y})," w : ",sess.run(W)," b: ",sess.run(b))
    plt.plot(plot_data['batch_num'],plot_data['loss'])
    plt.show()
    plt.plot(train_X, train_Y, 'ro', label='original data')
    plt.plot(train_X,train_X*sess.run(W)+sess.run(b))
    plt.show()
    print(sess.run(z,feed_dict={X:5}))











