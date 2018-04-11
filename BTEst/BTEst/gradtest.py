import tensorflow as tf

x1_data = [[1,10], [2,20], [9,90]]
y_data = [[1.1], [1.9],[ 3.2]]


x1 = tf.placeholder(tf.float32, name='x1')
y = tf.placeholder(tf.float32, name='y')

# 모델 (tf.Variable을 이용해 값이 바뀔수 있도록 한다. 여기선 tf.random_uniform을 이용해 초기화)
w = tf.Variable([[9.0] , [1.0]], name='w')
b = tf.Variable([0.0 ], name='b')

# 가설은 linear regression model을 표현. y = w * x + b
# w 와 x 가 행렬이 아니므로 tf.matmul 이 아니라 기본 곱셈 기호를 사용했습니다. (by golbin)
hypothesis = tf.add(tf.matmul( x1 , w ) ,b )

cost = tf.reduce_mean(  hypothesis  )


res  = tf.keras.backend.gradients( cost , b)[0]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result =sess.run([res] , feed_dict={x1: x1_data, y: y_data})
    print(result)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)



