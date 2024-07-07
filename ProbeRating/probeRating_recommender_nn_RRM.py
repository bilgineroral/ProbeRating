"""
Created on Sat November 30 21:14:09 2019

@author: Shu Yang

"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from scipy.stats import spearmanr
import numpy as np
import sys
import os
import h5py
import time

lossIdx = int(sys.argv[1])
optimizerIdx = int(sys.argv[2])
lrate = float(sys.argv[3])
epochsNum = int(sys.argv[4])
batchesPerEpoch = int(sys.argv[5])
numFolds = int(sys.argv[6])
l1weight = float(sys.argv[7]) 
l2weight = float(sys.argv[8])
isPermuteValid = int(sys.argv[9]) 
matFile = sys.argv[10]
p2vFile = sys.argv[11]    
normalizeFeature = int(sys.argv[12])  #0 or 1
activationFunc = sys.argv[13]
dropoutRate = float(sys.argv[14]) 
plateauPatience = int(sys.argv[15])   
earlyStopPatience = int(sys.argv[16])   
batchSize = int(sys.argv[17]) 

if p2vFile=='0':
    isP2V='0'
else:
    isP2V='1'

if lossIdx==1:
    myLoss='mean_squared_error'
elif lossIdx==2:
    myLoss='mean_absolute_percentage_error'
elif lossIdx==3:
    myLoss='mean_squared_logarithmic_error'
elif lossIdx==4:
    myLoss='logcosh'

sysuffix=os.getppid()   #get parent .sh process's PID
namePrefix='probeRating_nn_'+'rrm_p2vFT'+isP2V
syspec=namePrefix+'_noPermuteValid'+sys.argv[9]+'_loss'+sys.argv[1]+'_optimizer'+sys.argv[2]+'_lr'+sys.argv[3]+'_epochs'+sys.argv[4]+'_batchesPerEpoch'+sys.argv[5]+'_folds'+sys.argv[6]+'_ElasticNetAlpha'+sys.argv[7]+'_ElasticNetL1ratio'+sys.argv[8]+'_normalizeFeature'+sys.argv[12]+'_'+sys.argv[13]+'_dropOutRate'+sys.argv[14]+'_plateauPatience'+sys.argv[15]+'_earlyStopPatience'+sys.argv[16]+'_batchSize'+sys.argv[17]
dateTime = time.strftime("%Y_%m_%d-%H_%M_%S")

# Load data
dictData=h5py.File(matFile, 'r')
label=np.array(dictData['Y']).T 
RNAf=np.array(dictData['D']).T
protf=np.array(dictData['P']).T

if p2vFile!='0':
    protf=np.genfromtxt(p2vFile, delimiter=',')

# Tmp: normalize the input, only needed for SELU activation function
if normalizeFeature==1:
    RNAf=np.array(RNAf, dtype=np.float64)   #same as RNAf=RNAf.astype(np.float64) or RNAf=RNAf.astype(float) 
    RNAf-=RNAf.mean(axis=0)
    RNAf /=RNAf.std(axis=0)
    protf=protf.astype(np.float64)
    protf-=protf.mean(axis=0)
    protf /=protf.std(axis=0)

print("RNAf.shape:", RNAf.shape)
print("protf.shape:", protf.shape)
print("label.shape:", label.shape)
print("\n") 

# Initialize global vars
np.random.seed(0)   
proteinNum=len(protf)

# make data (no generator)

ridx=np.random.permutation(proteinNum) 
protfp=protf[ridx] 
labelp=label[:][:,ridx]

print("protfp.shape:", protfp.shape)
print("labelp.shape:", labelp.shape)
print("\n")

k=numFolds    
fold_ix=np.rint(np.linspace(0,proteinNum,k+1))
fold_ix=fold_ix.astype(int)

def correlation_coefficient_loss(y_true, y_pred):
    '''
    Use K.epsilon() == 10^-7 to avoid divide by zero error    
    '''
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.maximum(K.sum(K.square(xm)), K.epsilon()), K.maximum(K.sum(K.square(ym)), K.epsilon())))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(1 - r)

class Kron(Layer):
    """Merge Layer that mimic Kronecker product on a list of 2 inputs
    """
    def __init__(self, axis=-1, **kwargs):
        """
        **kwargs: standard layer keyword arguments.
        """
        super(Kron, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        pass

    def call(self, inputs):
        """
        Do Kronecker product on the last axis for the 2 input tensors. Note inputs tensors should have equal dimension for axis=0 case (ie. batchsize should be equal). 
        
        Alternatively, if inputs tensors have equal dimensions, can also use the implementation in outer_product() function below. 
        """
        output = K.repeat_elements(inputs[0], K.int_shape(inputs[1])[1], -1)
        inputs1_tiled = K.tile(inputs[1], [1, K.int_shape(inputs[0])[1]])
        return output * inputs1_tiled
        
    @staticmethod    
    def outer_product(inputs):
        """
        use the implementation in _merge_function() for outer product, since it is more general. This outer_product() function can only deal with inputs of 2 tensors with equal dimensions
        """
        inputs0, inputs1 = inputs
        batchSize = K.shape(inputs0)[0]
        outerProduct = inputs0[:, :, np.newaxis]*inputs1[:, np.newaxis, :]
        outerProduct = K.reshape(outerProduct, (batchSize, -1))
        return outerProduct    

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape)!=2:
            raise ValueError('A `Kronecker` layer should be called on a list of 2 inputs.')
        output_shape=list(input_shape[0])
        shape=list(input_shape[1])
        if output_shape[self.axis] is None or shape[self.axis] is None:
            output_shape[self.axis]=None
        output_shape[self.axis] *= shape[self.axis]
        return tuple(output_shape)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "axis": self.axis}


def kronecker(inputs, **kwargs):
    """Functional interface to the `Kron` layer.
    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the kronecker product of the inputs.
    """
    return Kron(**kwargs)(inputs)


def get_optimizer(optimizerIdx=2):
    if optimizerIdx==1:
        return optimizers.RMSprop(learning_rate=lrate, rho=0.9, epsilon=None) 
    elif optimizerIdx==2:
        return optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999, amsgrad=False)          


# Do k-fold CV with neural network 
CCs=np.zeros((proteinNum,2)) # correlation coefficients, 2 columns: pcc, pvalue of pcc | scc, pvalue of scc
CCs=np.concatenate([ridx.reshape((-1,1), order='F'), CCs], axis=1)
CCs1=CCs.copy()

print("CCs.shape:", CCs.shape)
print("CCs1.shape:", CCs1.shape)
print("\n")

for fold in range(k):
    test_ix=np.arange(fold_ix[fold],fold_ix[fold+1])
    train_ix=np.setdiff1d(np.arange(proteinNum), test_ix)

    # selecting training and testing proteins 
    protTrain=protfp[train_ix]
    protTest=protfp[test_ix]
    intensityTrain=labelp[:,train_ix]
    intensityTest=labelp[:,test_ix]

    print("protTrain.shape:", protTrain.shape)
    print("protTest.shape:", protTest.shape)
    print("intensityTrain.shape:", intensityTrain.shape)
    print("intensityTest.shape:", intensityTest.shape)
    print("\n")
    
    # make nonparam transformation. Change the task from intensity prediction to similarity prediction
    # protTrain has 7 proteins
    YTY=np.dot(intensityTrain.T, intensityTrain) # 7x50 * 50x7 = 7x7, cosine similarity of RBPs
    YTD=np.dot(intensityTrain.T, RNAf) # 7x50 * 50x1024 = 7x1024, compressed RNA matrix

    print("YTY.shape:", YTY.shape)
    print("YTD.shape:", YTD.shape)
    print("\n")

    # make input to nn
    rnaNum=YTD.shape[0] # 7 RNA
    protTrainIN=protTrain.repeat(rnaNum, axis=0) # 7 protein, 7 RNA => 49 pairs
    protTestIN=protTest.repeat(rnaNum, axis=0) # 8 protein, 7 RNA => 56 pairs
    similarityTrainIN=YTY.reshape((-1,1), order='F') 
    protTrainNum=protTrain.shape[0] # 7 protein
    protTestNum=protTest.shape[0] # 8 protein

    rnaTrainIN=np.tile(YTD,(protTrainNum,1)) # 7 protein, 7 RNA => 49 pairs
    rnaTestIN=np.tile(YTD,(protTestNum,1)) # 8 protein, 7 RNA => 56 pairs

    print("similarityTrainIN.shape:", similarityTrainIN.shape)
    print("protTrainNum:", protTrainNum)
    print("protTestNum:", protTestNum)
    print("\n")
    

    # permute training data 
    if isPermuteValid==1:
        ridx1=np.random.permutation(similarityTrainIN.shape[0])
        similarityTrainIN=similarityTrainIN[ridx1]
        protTrainIN=protTrainIN[ridx1]
        rnaTrainIN=rnaTrainIN[ridx1]

    # set up model
    if activationFunc=='selu':
        myInitializer="lecun_normal"
    elif activationFunc=='tanh':
        myInitializer="glorot_uniform"
    
    tensorBoardDir="../deepNN/%s-pid%d/%s/fold%d"%(dateTime, sysuffix, namePrefix, fold)   #/tensorBoardLog
  
    if not os.path.exists(tensorBoardDir):
        os.makedirs(tensorBoardDir)    
    checkPtFile='../deepNN/%s-pid%d/p2v-fold%d.hdf5'%(dateTime, sysuffix, fold)
    scriptInputArgs='../deepNN/%s-pid%d/scriptInputArgs.txt'%(dateTime, sysuffix)
    with open(scriptInputArgs,'w') as textFile:
        print(syspec, file=textFile)

    protTensor=Input(shape=(protTrainIN.shape[1],), name='FastProt')
    if activationFunc=='selu':
        x1=layers.AlphaDropout(dropoutRate)(protTensor)
    else:
        x1=layers.Dropout(dropoutRate)(protTensor)

    x1=layers.BatchNormalization()(x1)
    x1=layers.Dense(units=32, activation=activationFunc, kernel_initializer=myInitializer, kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01))(x1)
    x1=layers.BatchNormalization()(x1)
    
    rnaTensor=Input(shape=(rnaTrainIN.shape[1],), name='FastRNA')
    if activationFunc=='selu':
        x2=layers.AlphaDropout(dropoutRate)(rnaTensor)
    else:
        x2=layers.Dropout(dropoutRate)(rnaTensor)
   
    x2=layers.BatchNormalization()(x2)
    x2=layers.Dense(units=32, activation=activationFunc, kernel_initializer=myInitializer, kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01))(x2)   
    x2=layers.BatchNormalization()(x2)
    merged=layers.dot([x1, x2], -1)    
    similarity=layers.Dense(units=1, kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(merged) 
    network1=models.Model([protTensor, rnaTensor], similarity) 
    network1.compile(optimizer=get_optimizer(optimizerIdx=optimizerIdx), loss=myLoss, metrics=[correlation_coefficient_loss]) 
    callbacksList=[ModelCheckpoint(filepath=checkPtFile, verbose=0, monitor="val_loss", save_best_only=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=plateauPatience, min_lr=0.000001), EarlyStopping(monitor="val_loss", patience=earlyStopPatience), TensorBoard(tensorBoardDir, histogram_freq=0, embeddings_freq=0)] 

    # fit model
    # no-generator version:
    protTrainIN = tf.convert_to_tensor(protTrainIN, dtype=tf.float32)
    rnaTrainIN = tf.convert_to_tensor(rnaTrainIN, dtype=tf.float32)
    similarityTrainIN = tf.convert_to_tensor(similarityTrainIN, dtype=tf.float32)

    print("protTestIN.shape:", protTestIN.shape)
    print("rnaTestIN.shape:", rnaTestIN.shape)

    history=network1.fit([protTrainIN, rnaTrainIN], similarityTrainIN, batch_size=batchSize, epochs=epochsNum, verbose=0, callbacks=callbacksList, validation_split=0.1, shuffle=True)

    # evaluate model 
    predictedSimilarity=network1.predict([protTestIN, rnaTestIN]) # ()
    predictedSimilarity=predictedSimilarity.reshape((rnaNum,-1),order='F')  
    print("predictedSimilarity.shape:", predictedSimilarity.shape)
    # option1: Weighted sum reconstruction

    print("intensityTrain.shape:", intensityTrain.shape)
    print("********************************")
    intensityPred=np.dot(intensityTrain, predictedSimilarity)
    # option2:  Moore-Penrose pseudo inverse reconstruction:
    intensityPred1=np.dot(np.linalg.pinv(intensityTrain.T), predictedSimilarity)

    print("intensityPred.shape:", intensityPred.shape)
    print("intensityPred1.shape:", intensityPred1.shape)

    for i in range(test_ix.shape[0]):
        CCs[test_ix[i], 1], CCs[test_ix[i], 2]=spearmanr(intensityPred[:,i], intensityTest[:,i])
        CCs1[test_ix[i], 1], CCs1[test_ix[i], 2]=spearmanr(intensityPred1[:,i], intensityTest[:,i])


# Results
print("\n\n### the number of folds in cross validation =",k) 
print("The average SCC, pvalue of SCC:")
print(np.mean(CCs[:,1:], axis=0))
print("The average SCC1, pvalue of SCC1:")
print(np.mean(CCs1[:,1:], axis=0))
