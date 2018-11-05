from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 5
state_size = 10
echo_step = 3
batch_size = 50
num_batches = total_series_length//batch_size

N=1000
measurement_num=100

F=10

f_max= N//F

delta =10


f_array=np.random.uniform(0,1,[1,measurement_num])

f_array_tf= tf.cast(tf.reshape(tf.constant(f_array),[1,measurement_num]), tf.float32)

sparse_max=5
num_classes = 2




def generate_sparse_vector(amplitude, time_profile):
    X = np.zeros([total_series_length,N])
    Y = np.zeros([total_series_length,N])
    for j in range(total_series_length):
        for i in range(sparse_max):
            if amplitude[j,i,0]>0:
                pos = int(np.floor(time_profile[j,i,0]*N))-1
                if pos<1:
                    pos=0
                X[j,pos]=amplitude[j,i,0]
                Y[j,pos]=1
                for k in range(delta):
                    pos_1=pos-k-1
                    pos_2=pos+k+1
                    if pos_1>=0:
                        Y[j,pos_1]=1
                    if pos_2<N:
                        Y[j,pos_2]=1
                        
    return X,Y
                
def tf_generate_sparse_vector(amplitude, time_profile):
    X = tf.zeros([batch_size,N,1])
    Y = tf.zeros([batch_size,N,1])
    
    
    for i in range(truncated_backprop_length):
        data1= amplitude[i]
        data2= time_profile[i]
        
        batch_vector=[]
        pos_vector=[]
        
        for j in range(batch_size):
            pos = tf.cast(tf.floor(data2[j,0]*N), tf.int32)-1
            sparse_vector = tf.one_hot(pos, N)
            if j==0:
                batch_vector=sparse_vector*data1[j,0]
                pos_vector=sparse_vector
            else:
                batch_vector= tf.concat([batch_vector,sparse_vector*data1[j,0]],axis=0)
                pos_vector= tf.concat([pos_vector,sparse_vector],axis=0)
        batch_vector = tf.reshape(batch_vector,[batch_size,N,1])
        pos_vector = tf.reshape(pos_vector,[batch_size,N,1])
        
        X = tf.concat([X,batch_vector],axis=2)
        Y = tf.concat([Y,pos_vector],axis=2)
        
        
    X = tf.reduce_sum(X,axis=2)
    Y = tf.reduce_max(Y,axis=2)
    return X,Y
    


def generatesingleM(amplitude, time_profile):
    aa = tf.matmul(time_profile,f_array_tf)
    amplitude_s= tf.reshape(tf.tile(amplitude,[1,measurement_num]),[batch_size,measurement_num])
    return tf.multiply(amplitude_s,tf.sin(2*np.pi*aa))

def generateMeasurement(amplitude, time_profile):
    
    aa = np.tensordot(time_profile,f_array,axes=[[2],[0]])
    
    amplitude_s= np.reshape(np.tile(amplitude,[1,1,measurement_num]),[total_series_length,sparse_max,measurement_num])
    
    return np.sum(np.multiply(amplitude_s,np.sin(2*np.pi*aa)),axis=1)
    

def generateData2():
    x = np.array(np.random.choice(2, [total_series_length,sparse_max], p=[0.3, 0.7]))
    amplitude = np.random.uniform(0,1,[total_series_length,sparse_max])
    amplitude = np.multiply(x,amplitude)
    
    time_profile = np.random.uniform(0,1,[total_series_length,sparse_max,1])
    amplitude= np.reshape(amplitude,[total_series_length,sparse_max,1])
    
    #print(amplitude,time_profile)
    
    

    a = generateMeasurement(amplitude,time_profile)
    
    
    b,c = generate_sparse_vector(amplitude,time_profile)

    return (a,b,c)



def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, measurement_num])
batchY_placeholder = tf.placeholder(tf.float32, [batch_size, N])
batchZ_placeholder = tf.placeholder(tf.float32, [batch_size, N])

init_state = tf.placeholder(tf.float32, [2, batch_size, state_size])

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
#inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, axis=1)
#labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes
cell = tf.contrib.rnn.BasicLSTMCell(state_size)

hidden_state = tf.zeros([batch_size, state_size])
current_state = tf.zeros([batch_size, state_size])
state = hidden_state, current_state


#states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, dtype=tf.float32)

losses=[]
predictions_series=[]
amplitudes = []
phases = []

beitrag=0
input_data=0

with tf.variable_scope("RNN"):
  for time_step in range(truncated_backprop_length):
    if time_step > 0: 
        tf.get_variable_scope().reuse_variables()
        input_data=input_data - beitrag
    else:
        input_data=batchX_placeholder

    # modify the state
    modified_state = state

    output, state = cell(input_data, modified_state)

    logits = tf.matmul(output, W2) + b2
    
    predictions = tf.nn.softmax(logits)
    
    amplitude_info = tf.reshape(predictions[:,0], [batch_size,1]);
    phase_info = tf.reshape(predictions[:,1], [batch_size,1]);
    
    
    beitrag = generatesingleM(amplitude_info, phase_info)
    

#    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_series[time_step])
#
    predictions_series.append(predictions)
    amplitudes.append(amplitude_info)
    phases.append(phase_info)
    
    
#    losses.append(loss)
    
    
amplitudes= tf.unstack(amplitudes)
phases= tf.unstack(phases)

sparses, positions= tf_generate_sparse_vector(amplitudes, phases)
#
diff_sparses= tf.reduce_mean(tf.reduce_sum(tf.square(sparses-batchY_placeholder), axis=1))

pos_sparses= tf.reduce_mean(tf.reduce_sum(tf.nn.relu(positions-batchZ_placeholder), axis=1))

sparsity = tf.reduce_mean(tf.reduce_sum(positions, axis=1))
    
current_state = state
    
total_loss = tf.reduce_mean(tf.reduce_sum(tf.square(input_data), axis=1))+0.05*pos_sparses

train_step = tf.train.AdagradOptimizer(0.1).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
    

    for epoch_idx in range(num_epochs):
        x,y = generateData()
        a,b,c = generateData2()
        
        _current_state = np.zeros((2,batch_size, state_size))

        print("New data, epoch", epoch_idx)
        
        loss_gen=0
        pos_loss=0
        diff_loss=0
        s_avg=0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batchX = a[start_idx:end_idx,:]
            batchY = b[start_idx:end_idx,:]
            batchZ = c[start_idx:end_idx,:]

            _total_loss, _train_step, _diff_sparses, _pos_sparses,_sparsity, p ,p2 = sess.run(
                [total_loss, train_step, diff_sparses, pos_sparses, sparsity, positions, phases],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    batchZ_placeholder:batchZ
                    #init_state:_current_state
                })
            loss_list.append(_total_loss)
            
#            print(p, p2)
#            
#            sdada
            
            diff_loss+= _diff_sparses
            pos_loss += _pos_sparses
            loss_gen+=_total_loss
            s_avg += _sparsity
            if batch_idx%100 == 0:
                print (loss_gen/(batch_idx+1), diff_loss/(batch_idx+1), pos_loss/(batch_idx+1), s_avg/(batch_idx+1))
        print ("Finishing epoch")    
        print (loss_gen/num_batches, diff_loss/num_batches, pos_loss/num_batches, s_avg/num_batches)
#            if batch_idx%100 == 0:
#                print("Step",batch_idx, "Loss", _total_loss)
#                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()