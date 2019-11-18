import pickle
import json
import tensorflow as tf
from tensorflow.keras import backend as K

# Wrapper for Keras CTC Loss
def ctcLoss(yTrue, yPred):    
    # Reshape the ground truth tensor into shape required by ctc_batch_cost().
    yTrueShape = K.shape(yTrue)
    yTrue = K.reshape(yTrue, shape=(yTrueShape[0], yTrueShape[1]))

    # Get the input sequence and label sequence length for each sample in the batch.
    hasTrueLables = K.clip(yTrue + 1, 0, 1)
    labelLength = K.sum(hasTrueLables, axis=1, keepdims=True)
    hasPredLabels = K.sum(yPred, axis=2)
    inputLength = K.sum(hasPredLabels, axis=1, keepdims=True)
    return K.ctc_batch_cost(yTrue, yPred, inputLength, labelLength)


# Calculate the log_probability for sequence model accuracy
def logProb(yTrue, yPred):    
    hasPredLabels = K.sum(yPred, axis=2)
    inputLength = K.sum(hasPredLabels, axis=1)
    # We need to call CTC_decode to take out blank/repeated characters
    outputPaths, logProbTensor = K.ctc_decode(yPred, inputLength)
    avgLogProb = K.sum(logProbTensor, axis=0, keepdims=False)
    return avgLogProb

# Convert a dense to Sparse Tensor
def denseToSparse(tensor):    
    # Get indexes of non-blank symbols (blank symbol is used for padding and is represented by -1).
    indexes = tf.where(tf.not_equal(tensor, tf.constant(-1, tensor.dtype)))
    values = tf.gather_nd(tensor, indexes)
    shape = tf.shape(tensor, out_type=tf.int64)
    return tf.SparseTensor(indexes, values, shape)


# Calculate editDistance with tf.edit_distance
def editDistance(yTrue, yPred):    
    # Reshape the ground truth tensor into the required shape.
    yTrueShape = K.shape(yTrue)
    yTrue = K.cast(K.reshape(yTrue, shape=(yTrueShape[0], yTrueShape[1])), dtype='int64')

    # Decode the predicted sequence.
    hasPredLabels = K.sum(yPred, axis=2)
    inputLength = K.sum(hasPredLabels, axis=1)
    # We need to call CTC_decode to take out blank/repeated characters
    # The Model using CTC_loss always need to use ctc_decode on inference
    outputPaths, logProbTensor = K.ctc_decode(yPred, inputLength)

    # Compute the normalized Levenshtein edit distance.
    trueSparseTensor = denseToSparse(yTrue)
    predSparseTensor = denseToSparse(outputPaths[0])
    editDistanceTensor = tf.edit_distance(predSparseTensor, trueSparseTensor)

    # Sum the edit distance for individual samples in the batch.
    distance = K.sum(editDistanceTensor, axis=0)
    return distance

def load_pickle(data_file):
    try:
        f = open(data_file,"rb")
        dict_iam = pickle.load(f)
        f.close()
        return dict_iam
    except FileNotFoundError:
        return None

def load_json(data_file):
    with open(data_file) as json_file:
        data = json.load(json_file)
        return data
    

class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        #self.writer.add_summary(logs['lr'], epoch)
        tf.summary.scalar('learning rate', data=logs['lr'], step=epoch)
        print('\t*******LearningRate:', K.eval(self.model.optimizer.lr))        
        # Add learning rate
        #logs.update({'lr': K.eval(self.model.optimizer.lr)})
        # Call default callback
        super().on_epoch_end(epoch, logs)