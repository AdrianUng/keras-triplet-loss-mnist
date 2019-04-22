from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model,load_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt

from keras.utils import plot_model


def triplet_euclidean_distance(vects):
    anch, pos, neg = vects
    anch_pos = K.sum(K.square(anch - pos), axis=1, keepdims=True)
    anch_neg = K.sum(K.square(anch - neg), axis=1, keepdims=True)

    return anch_pos - anch_neg


##############################################
def triplet_loss(y_true, y_pred):
    del y_true
    margin = 0.5  # 0.2
    operator1 = margin + y_pred
    operator2 = K.epsilon()

    return K.mean(K.maximum(operator1, K.epsilon()))


##############################################
def eucl_dist_output_shape(shapes):
    try:
        shape1, shape2 = shapes
    except:
        shape1, shape2, shape3 = shapes
    return (shape1[0], 1)


def create_triplet_pairs(x, digit_indices):
    anch, pos, neg = [], [], []

    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):

        for i in range(n):
            anch.append(x[digit_indices[d][i]])  # anchor
            pos.append(x[digit_indices[d][i + 1]])  # template from -> same class...
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes  # other class
            neg.append(x[digit_indices[dn][i]])  # template from  -> different clas...

            # # used for debugging -> visualizing Triplet pairs...
            # plt.subplot(131)
            # plt.imshow(x[digit_indices[d][i]])
            # plt.subplot(132)
            # plt.imshow(x[digit_indices[d][i+1]])
            # plt.subplot(133)
            # plt.imshow(x[digit_indices[dn][i]])
            # plt.show()

    return np.array(anch), np.array(pos), np.array(neg)


##############################################
def create_base_network(input_shape):
    """
    Base network to be shared (eq. to feature extraction).
    """
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


num_classes = 10
epochs = 100

train_flag = True


# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
input_shape = x_train.shape[1:]

# Create training + test triplet pairs...
# Training data
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]  # templates from train distribution
anchor_train, positive_train, negative_train = create_triplet_pairs(x_train, digit_indices)

# Validation data
digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]  # templates from test distribution
triplet_test_pairs = create_triplet_pairs(x_test, digit_indices)
anchor_val, positive_val, negative_val = triplet_test_pairs[0][:2000], \
                                         triplet_test_pairs[1][:2000], \
                                         triplet_test_pairs[2][
                                         :2000]  # for validation, grab the first 2000 templates
anchor_test, positive_test, negative_test = triplet_test_pairs[0][-100:], \
                                            triplet_test_pairs[1][-100:], \
                                            triplet_test_pairs[2][
                                            -100:]  # for validation, grab the last 2000 templates
if train_flag == True:
    # Network definition
    base_network = create_base_network(input_shape)
    plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)

    input_a = Input(shape=input_shape, name='input_anchor')
    input_p = Input(shape=input_shape, name='input_positive')
    input_n = Input(shape=input_shape, name='input_negative')
    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the THREE branches
    processed_a = base_network(input_a)
    processed_p = base_network(input_p)
    processed_n = base_network(input_n)

    distance = Lambda(triplet_euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_p, processed_n])

    model = Model(inputs=[input_a, input_p, input_n], outputs=distance)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    # train session
    rms = Adam(lr=0.0001)  # choose optimiser. RMS is good too!
    model.compile(
        loss=triplet_loss,  # contrastive_loss,
        optimizer=rms,
    )
    # Uses 'dummy' gt distances, of shape (NxM) where N is nr of input templates, M prediction size (this case M=1)
    dummy_gt_train = np.zeros((len(anchor_train), 1))
    dummy_gt_val = np.zeros((len(anchor_val), 1))

    H = model.fit(
        x=[anchor_train, positive_train, negative_train],
        y=dummy_gt_train,
        batch_size=128,
        epochs=epochs,
        validation_data=([anchor_val, positive_val, negative_val], dummy_gt_val))

    plt.plot(H.history['loss'], label='training loss')
    plt.plot(H.history['val_loss'], label='validation loss')
    plt.legend()
    plt.title('Train/validation loss')
    plt.show()

    model.save('triplet_mnist_v1_ep%d.hdf5' % epochs)

if train_flag == False:
    model = load_model('triplet_mnist_v1_ep100.hdf5',custom_objects={'triplet_loss':triplet_loss} )

# Grabbing the weights from the trained network
testing_embeddings = create_base_network(input_shape)  # creating an empty network
for layer_target, layer_source in zip(testing_embeddings.layers, model.layers[3].layers):
	weights = layer_source.get_weights()
	layer_target.set_weights(weights)
	del weights

# Test the network
nrs = np.arange(0, 50)  # first 50 instances...
differences = []
diff_baseline = []

for nr in nrs:
	plt.figure(figsize=(7, 2))
	anch, pos, neg = np.array(anchor_test[nr]), \
					 np.array(positive_test[nr]), \
					 np.array(negative_test[nr])

	anch_embeddings = testing_embeddings.predict(np.reshape(anch, (1, anch.shape[0], anch.shape[1])))
	pos_embeddings = testing_embeddings.predict(np.reshape(pos, (1, anch.shape[0], anch.shape[1])))
	neg_embeddings = testing_embeddings.predict(np.reshape(neg, (1, anch.shape[0], anch.shape[1])))

	a_p = np.sum(np.square(anch_embeddings - pos_embeddings))
	a_n = np.sum(np.square(anch_embeddings - neg_embeddings))

	diff_baseline.append([np.mean(np.square(anch - pos)), np.mean(np.square(anch - neg))])
	print([np.mean(np.square(anch - pos)), np.mean(np.square(anch - neg))])
	differences.append([a_p, a_n])
	print([a_p, a_n])
	print('#########')

	plt.subplot(131)
	plt.imshow(anchor_test[nr])
	plt.subplot(132)
	plt.imshow(positive_test[nr])
	plt.subplot(133)
	plt.imshow(negative_test[nr])
	plt.suptitle('AP=%.2f   AN=%.2f' % (a_p, a_n))
	plt.show()
print(diff_baseline)
print(differences)

pass
