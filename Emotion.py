import tensorflow as tf
import numpy as np
import os
import re

class emotion(object):
    def __init__(self, checkpoint_directory, steps):
        self.image_size = 48
        self.num_labels = 7
        self.num_channels = 1
        #self.batch_size = 16
        self.batch_size = 16
        self.kernal_1_size = 9
        self.kernal_2_size = 5
        self.depth1 = 64
        self.depth2 = 128
        self.fully_connected_depth1 = 1024
        self.fully_connected_depth2 = 512
        self.num_steps= 20000
        self.x_train = np.ndarray(shape=(20000, 2304))
        self.x_valid = np.ndarray(shape=(7945, 2304))
        self.x_test = np.ndarray(shape=(7943, 2304))
        self.y_train = np.ndarray(shape=(20000))
        self.y_valid = np.ndarray(shape=(7945))
        self.y_test = np.ndarray(shape=(7943))
        if not checkpoint_directory:
            self.checkpoint_dir = r"C:\\Users\\Joy.DESKTOP-M53NCFS\\Documents\\GitHub\\Emotion-recognizer\\checkpoints"
        else:
            self.checkpoint_dir = checkpoint_directory
        if not steps:
            self.num_steps = 20000
        else:
            self.num_steps = steps
    def reshape(self, data, labels):
        data = data.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)
        labels = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return data,labels

    def preprocess_datasets(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        self.x_train, self.y_train = self.reshape(x_train, np.array(y_train).astype(np.float32))
        self.x_valid, self.y_valid = self.reshape(x_valid, np.array(y_valid).astype(np.float32))
        self.x_test, self.y_test = self.reshape(x_test, np.array(y_test).astype(np.float32))

    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))) / predictions.shape[0]

    def cnn(self, data):

        conv = tf.nn.conv2d(data, self.w_1, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + self.b_1)
        pool = tf.nn.max_pool(hidden, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        norm = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        conv = tf.nn.conv2d(norm, self.w_2, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + self.b_2)
        pool = tf.nn.max_pool(hidden, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        norm = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        list_shape = norm.get_shape().as_list()
        reshape = tf.reshape(pool, [list_shape[0], list_shape[1] * list_shape[2] * list_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, self.w_3) + self.b_3)
        hidden = tf.nn.relu(tf.matmul(hidden, self.w_4) + self.b_4)
        dropout = tf.nn.dropout(hidden, self.dropout_prob)
        return tf.matmul(hidden, self.w_5) + self.b_5

    def model(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.tf_x = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size, self.num_channels))
            self.tf_y = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
            self.dropout_prob = tf.placeholder(tf.float32)

            self.w_1 = tf.Variable(tf.truncated_normal([self.kernal_1_size, self.kernal_1_size, self.num_channels, self.depth1], stddev=0.01))
            self.b_1 = tf.Variable(tf.zeros([self.depth1]))
            self.w_2 = tf.Variable(tf.truncated_normal([self.kernal_2_size, self.kernal_2_size, self.depth1, self.depth2], stddev=0.01))
            self.b_2 = tf.Variable(tf.constant(1.0, shape=[self.depth2]))
            self.w_3 = tf.Variable(tf.truncated_normal([self.image_size//4 * self.image_size//4 * self.depth2, self.fully_connected_depth1], stddev=0.01))
            self.b_3 = tf.Variable(tf.constant(1.0, shape=(self.fully_connected_depth1,)))
            self.w_4 = tf.Variable(tf.truncated_normal([self.fully_connected_depth1, self.fully_connected_depth2], stddev=0.1))
            self.b_4 = tf.Variable(tf.constant(1.0, shape=(self.fully_connected_depth2,)))
            self.w_5 = tf.Variable(tf.truncated_normal([self.fully_connected_depth2, self.num_labels], stddev=0.04))
            self.b_5 = tf.Variable(tf.constant(1.0, shape=[self.num_labels]))

            self.logits = self.cnn(self.tf_x)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_y, logits=self.logits))

            self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

            self.train_pred = tf.nn.softmax(self.logits)
            tf.summary.histogram('weights_1', self.w_1)
            tf.summary.histogram('weights_2', self.w_2)
            tf.summary.histogram('weights_3', self.w_3)
            tf.summary.histogram('weights_4', self.w_4)

            kernel_transposed = tf.transpose (self.w_1, [3, 0, 1, 2])
            tf.summary.image('conv1/filters', kernel_transposed)

            tf.summary.scalar('loss', self.loss)
            self.merged = tf.summary.merge_all()

    def load_checkpoints(self, bool_print):

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(self.checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)",ckpt_name)).group(0))
            if bool_print:
                print("Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print("Failed to find a checkpoint")
            return False, 0


    def train_test_validate(self):
        self.model()
        with tf.Session(graph=self.graph) as self.session:
            train_writer=tf.summary.FileWriter(r"C:\Users\Joy.DESKTOP-M53NCFS\Documents\GitHub\Emotion-recognizer\logs", self.graph)
            self.saver = tf.train.Saver()
            if not os.path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
                print ("\nCreated a directory for checkpoint")
            else:
                print ("\nDirectory for checkpoint already existing")

            bool,counter = self.load_checkpoints(True)

            if not bool:
                tf.global_variables_initializer().run()

            print ('Intialized')

            for step in range(counter, self.num_steps):
                offset1 = (step * self.batch_size) % (self.y_train.shape[0] - self.batch_size)
                batch_data1 = self.x_train[offset1:(offset1 + self.batch_size), :, :, :]
                batch_labels1 = self.y_train[offset1:(offset1 + self.batch_size), :]
                feed_dict1 = {self.tf_x : batch_data1, self.tf_y : batch_labels1, self.dropout_prob : 0.7}
                #print (self.logits.eval(feed_dict1))
                _,summary, l1, predictions1 = self.session.run([self.optimizer, self.merged, self.loss, self.train_pred], feed_dict=feed_dict1)
                train_writer.add_summary(summary, step)
                train_writer.flush()

                offset2 = (step * self.batch_size) % (self.y_valid.shape[0] - self.batch_size)
                batch_data2 = self.x_valid[offset2:(offset2 + self.batch_size), :, :, :]
                batch_labels2 = self.y_valid[offset2:(offset2 + self.batch_size), :]
                feed_dict2 = {self.tf_x : batch_data2, self.tf_y : batch_labels2, self.dropout_prob : 1.0}
                _, l2, predictions2 = self.session.run([self.optimizer, self.loss, self.train_pred], feed_dict=feed_dict2)

                offset3 = (step * self.batch_size) % (self.y_test.shape[0] - self.batch_size)
                batch_data3 = self.x_test[offset3:(offset3 + self.batch_size), :, :, :]
                batch_labels3 = self.y_test[offset3:(offset3 + self.batch_size), :]
                feed_dict3 = {self.tf_x : batch_data3, self.tf_y : batch_labels3, self.dropout_prob : 1.0}
                _, l3, predictions3 = self.session.run([self.optimizer, self.loss, self.train_pred], feed_dict=feed_dict3)
                if step%50==0:
                    print ('Minibatch loss at step %d: %f' % (step, l1))
                    print ('Minibatch accuracy: %.1lf%%' % self.accuracy(predictions1, batch_labels1))
                    print ('Validation accuracy: %.1lf%%' % self.accuracy(predictions2, batch_labels2))
                    print ('Test accuracy: %.1lf%%' % self.accuracy(predictions3, batch_labels3))

                if step%100==0:
                    self.saver.save(self.session, self.checkpoint_dir + "\checkpoints", global_step=step)

    def evaluate(self, data):
        self.batch_size = 1
        self.model()
        with tf.Session(graph=self.graph) as self.session:
            train_writer=tf.summary.FileWriter('F:\Python\Emotion\logs',self.graph)
            self.saver = tf.train.Saver()

            bool,counter = self.load_checkpoints(False)

            print ("\n")

            if not bool:
                tf.global_variables_initializer().run()

            data = data.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)
            x = self.session.run([self.train_pred], feed_dict={self.tf_x : data, self.dropout_prob: 0.0})
            x = np.array(x).reshape(7)
            emotion_dict = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
            self.batch_size = 16
            emotion_percentage = np.array([y/np.sum(x)*100 for y in x])
            emotion_percentage = emotion_percentage.astype(np.int32)
            emotion_with_percentage = []
            for x  in emotion_dict:
                emotion_with_percentage.append((emotion_dict[x],emotion_percentage[x]))
            emotion_with_percentage.sort(key=lambda x: x[1])
            print (emotion_with_percentage)
            return emotion_with_percentage[-1]
