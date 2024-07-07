import numpy as np
import h5py
import os, subprocess
from keras.utils import custom_object_scope
import tensorflow as tf
from tensorflow.keras import backend as K

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


def predict_proberating(protein_path, RNAf, prior):
    embeddings_dir = "out"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    fastbioseq_model_path = "./ProbeRating/sample_data/trained_FastBioseq_models/sample1_model"
    
    train_path = "./ProbeRating/sample_data/sample1.fa"
    protein_feature_vector_path = os.path.join(embeddings_dir, "protein_embeddings.csv")
    subprocess.run(["python", "./ProbeRating/train_FastBioseq.py", train_path, fastbioseq_model_path, "10", "3", "2"])
    subprocess.run(["python", "./ProbeRating/genVec.py", fastbioseq_model_path, protein_path, protein_feature_vector_path, "2"])
    
    model_path = "./deepNN/2024_07_07-17_50_40-pid4174/p2v-fold0.hdf5"

    with custom_object_scope({'correlation_coefficient_loss': correlation_coefficient_loss}):
        model = tf.keras.models.load_model(model_path)

    protein_vector = np.genfromtxt(protein_feature_vector_path, delimiter=',').reshape(1, -1)

    YTD = np.dot(prior.T, RNAf)
    
    rna_num = YTD.shape[0] # 15 in training data

    protein_vector_in = protein_vector.repeat(rna_num, axis=0) # (15, 10)
    rna_vector_in = np.tile(YTD, (1, 1)) # (15, 1024)
    
    predicted_similarity = model.predict([protein_vector_in, rna_vector_in])
    predicted_similarity = predicted_similarity.reshape((rna_num, -1), order='F') # (15, 1)
    
    # Option 1: Weighted sum reconstruction
    intensity_pred1 = np.dot(prior, predicted_similarity).flatten()

    # Option 2: Moore-Penrose pseudo inverse reconstruction
    intensity_pred2 = np.dot(np.linalg.pinv(prior.T), predicted_similarity)
    return intensity_pred1, intensity_pred2


if __name__ == "__main__":
    protein_path = "./ProbeRating/sample_data/prot.fa"
    mat_file_path = "./ProbeRating/sample_data/sample4.mat" # File containing intensity scores of RBPs
    dictData=h5py.File(mat_file_path, 'r')

    RNAf=np.array(dictData['D']).T # RNA features
    prior=np.array(dictData['Y']).T # prior binding intensity scores

    intensity_pred1, intensity_pred2 = predict_proberating(protein_path, RNAf, prior)

    print("Predicted binding intensities of the given protein to 50 RNA probes in the training data:")
    print("Predicted binding intensities using weighted sum reconstruction method:", intensity_pred1)
    print()
    print("Predicted binding intensities using pseudo-inverse reconstruction method:", intensity_pred2)

    max_val_1_idx = np.argmax(intensity_pred1)
    max_val_1 = intensity_pred1[max_val_1_idx]

    max_val_2_idx = np.argmax(intensity_pred2)
    max_val_2 = intensity_pred2[max_val_2_idx]
    
    print("The probe with the highest predicted binding intensity using weighted sum reconstruction method is probe", max_val_1_idx, "with a predicted binding intensity of", max_val_1)
    print("The probe with the highest predicted binding intensity using pseudo-inverse reconstruction method is probe", max_val_2_idx, "with a predicted binding intensity of", max_val_2)
