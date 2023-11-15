import abc
import random
from abc import ABCMeta
from pathlib import Path
from typing import List

import keras.backend as kbe
import numpy as np
import pandas as pd
from keras import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
import keras.preprocessing.image as kpimg
from keras.src.metrics import F1Score
from keras.models import load_model
from tensorflow.math import confusion_matrix
from tensorflow.keras.metrics import Accuracy, F1Score
from tqdm import tqdm
from face_detection import utils

# this is currently set manually based on eyeballing the positive P75 and negative P25 values (from Validate)
DISSIMILARITY_THRESHOLD:float = 0.65

def euclidian_distance(vectors:List) -> float:
    v1, v2 = vectors
    return kbe.sqrt(kbe.maximum(kbe.sum(kbe.square(v1 - v2), axis=1, keepdims=True), kbe.epsilon()))


def contrastive_loss(y_true, D) -> float:
    margin = 1.0
    return kbe.mean((y_true * kbe.square(D)) + (1 - y_true) * kbe.maximum((margin - D), 0))


def accuracy(y_true, y_pred) -> float:
    return kbe.mean(kbe.equal(y_true, kbe.cast(y_pred < DISSIMILARITY_THRESHOLD, y_true.dtype)))


class SiameseNNDataLoader(metaclass=ABCMeta):

    @abc.abstractmethod
    def load(self, train_test_split: float, reload: bool):
        raise NotImplemented()

    @abc.abstractmethod
    def get_input_shape(self) -> tuple:
        raise NotImplemented()

    @abc.abstractmethod
    def generate_pairs_and_labels(self, use_train: bool):
        raise NotImplemented()


class ATTSiameseNNDataLoader(SiameseNNDataLoader):
    '''
    utility class to load and vend images from training datasets
    '''

    def __init__(self, root_dir: Path, max_positive_pairs_per_image: int = 5,
                 max_negative_pairs_per_image: int = 5, pair_samples_path:Path = None) -> None:
        '''
        Intialize an instance with...
        :param root_dir: pathlib Path to the directory hosting the class subdirectories
        :param max_positive_pairs_per_image: how many same-class pairs to make per image
        :param max_negative_pairs_per_image: how many different-class pairs to make per image
        :param pair_samples_path: if present, will output side-by-side image pairs for visual debugging
        '''
        super().__init__()
        self.img_train = None
        self.class_train = None
        self.img_test = None
        self.class_test = None
        assert root_dir is not None and root_dir.is_dir()
        self.root_dir = root_dir
        self.max_positive_pairs_per_image = max_positive_pairs_per_image
        self.max_negative_pairs_per_image = max_negative_pairs_per_image
        self.pair_samples_path = pair_samples_path

    def load(self, train_test_split: float = 0.75, reload: bool = True) -> tuple[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ]:
        '''
        loads the image data into memory (needs to be refactored to leave image data on disk to handle big training sets)
        :param train_test_split: ratio of training samples to test (1.0 = all train)
        :param reload: force reload if the data is already loaded
        :return: (tuple of training images and training labels) , (tuple of test images and test labels)
        '''
        assert train_test_split >= 0.0 and train_test_split <= 1.0
        print(f"loading data from {self.root_dir}")

        if self.img_train is None or reload:
            class_dirs: List = [x for x in self.root_dir.iterdir() if x.is_dir()]
            num_classes: int = len(class_dirs)

            # choose the classes in train and test
            num_train_classes: int = int(num_classes * train_test_split)
            train_class_dirs: List = random.sample(class_dirs, num_train_classes)
            test_class_dirs: List = [d for d in class_dirs if d not in train_class_dirs]

            img_train, class_train, img_test, class_test = [], [], [], []
            for idx, dir in tqdm(enumerate(class_dirs)):
                for f in sorted([x for x in dir.iterdir() if x.is_file()]):
                    img = kpimg.load_img(str(f), color_mode='grayscale')
                    img = kpimg.img_to_array(img).astype('float32') / 255
                    img = img.reshape(img.shape[0], img.shape[1], 1)
                    if dir in train_class_dirs:
                        img_train.append(img)
                        class_train.append(train_class_dirs.index(dir))
                    else:
                        img_test.append(img)
                        class_test.append(test_class_dirs.index(dir))

            self.img_train = np.array(img_train)
            self.class_train = np.array(class_train)
            self.img_test = np.array(img_test)
            self.class_test = np.array(class_test)

        # returns list of images (1d, each class sequentially) with matching class labels
        return (self.img_train, self.class_train), (self.img_test, self.class_test)

    def get_input_shape(self) -> tuple:
        return self.img_train.shape[1:]

    def generate_pairs_and_labels(self, use_train: bool = True, ) -> tuple:
        '''
        creates a list of pairs and associated labels
        :param use_train: True for training pairs, False for test/validation pairs
        :return: NDArray of image pair tuples (one face each) and a matching NDArray of labels (1 = same class, 0 = different class)
        '''
        print(f"generating pairs...")
        pairs = []
        labels = []

        # loop through each image
        if use_train:
            X = self.img_train
            Y = self.class_train
        else:
            X = self.img_test
            Y = self.class_test

        rng = np.random.default_rng()
        idxs = range(X.shape[0])
        for idx in tqdm(idxs):
            same_class_idxs = np.where(Y == Y[idx])[0]
            same_class_idxs = [xidx for xidx in same_class_idxs if xidx != idx]
            diff_class_idxs = np.where(Y != Y[idx])[0]

            # randomly select 5 positive pairings from remaining class images
            pos_pair_idxs = rng.choice(
                same_class_idxs,
                self.max_positive_pairs_per_image if self.max_positive_pairs_per_image <= len(same_class_idxs) else len(same_class_idxs),
                replace=False
            )
            for i in pos_pair_idxs:
                pairs.append((X[idx], X[i]))
                labels.append(1.0)
            # output the last pair from above for visual QA
            if self.pair_samples_path is not None:
                utils.side_by_side_from_arrays(X[idx], X[i], Path(self.pair_samples_path,f"{idx}_{i}_1.jpg"))

            # randomly select 5 negative pairings from other classes
            neg_pair_idxs = rng.choice(diff_class_idxs, self.max_negative_pairs_per_image, replace=False)
            for i in neg_pair_idxs:
                pairs.append((X[idx], X[i]))
                labels.append(0.0)
            # output the last pair from above for visual QA
            if self.pair_samples_path is not None:
                utils.side_by_side_from_arrays(X[idx], X[i], Path(self.pair_samples_path,f"{idx}_{i}_0.jpg"))
        print(f" .. done; generated {len(labels)} pairs")
        return np.array(pairs), np.array(labels)


class SiameseNeuralNetTrainer():
    '''
    wraps Keras model to train a SiameseNN
    '''

    def __init__(self, loader: SiameseNNDataLoader, model:Model = None) -> None:
        '''
        Creates instance with...
        :param loader: SiameseNNDataLoader instance to generate train and test pairs
        :param model: if None, will train a model from scratch; else will load from file (for validation)
        '''
        super().__init__()
        assert loader is not None
        self.loader = loader
        self.loader.load(reload=False)

        self.model = None
        self.model_trained = False
        if model is not None:
            self.model = model
            self.model_trained = True

    def _create_common_model(self):
        '''
        creates the core convolutional model that will feed the euclidean distance
        :return: Keras model
        '''
        common_model = Sequential(name='common')
        common_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=self.loader.get_input_shape()))
        common_model.add(MaxPooling2D())
        common_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        common_model.add(Flatten())
        common_model.add(Dense(units=128, activation='sigmoid'))
        return common_model

    def assemble(self, shape:tuple):
        '''
        creates the Siamese NN Keras model
        :param shape: the image shape to be trained on
        '''
        print(f"assempling NN with shape {shape}")
        if model is None:
            input_A = Input(shape=shape)
            input_B = Input(shape=shape)
            common_model = self._create_common_model()
            output_A = common_model(input_A)
            output_B = common_model(input_B)
            distance = Lambda(euclidian_distance, output_shape=(1,))([output_A, output_B])
            self.model = Model(inputs=[input_A, input_B], outputs=distance)

    def train(self, epochs: int = 10, batch_size: int = 128) -> Model:
        '''
        trains the model on pairs generate by the data loader
        :param epochs: number of training epochs
        :param batch_size: self explanatory
        :return: trained model
        '''
        print(f"training model...")
        training_pairs, training_labels = self.loader.generate_pairs_and_labels()
        print(f" .. compiling model")
        self.model.compile(
            loss=contrastive_loss,
            optimizer='adam',
            metrics=[accuracy],
            # metrics=[Accuracy()],
            # metrics=F1Score(),
        )
        print(f" .. fitting model")
        v_pairs, v_labels = self.loader.generate_pairs_and_labels(use_train=False)
        t1 = training_pairs[:, 0]
        t2 = training_pairs[:, 1]
        self.model.fit(
            [t1, t2],
            training_labels,
            batch_size=batch_size,
            epochs=epochs,
            # validation_split=0.1,
            validation_data=([v_pairs[:,0],v_pairs[:,1]], v_labels),
        )
        return self.model

    def validate(self ) -> float:
        '''
        Utility method that outputs several training result attributes
        :return: F1Score
        '''
        print("validating model")
        test_pairs, test_labels = self.loader.generate_pairs_and_labels(use_train=False)
        res = self.model.evaluate(
            [test_pairs[:,0],test_pairs[:,1]],
            test_labels,
            batch_size=128,
            return_dict=True,
        )
        print(f"results from evaluate: {res}")

        y_pred = model.predict(
            [test_pairs[:,0], test_pairs[:,1]],
            batch_size=32,
        )
        # for idx, pair in enumerate(test_pairs):
        #     utils.side_by_side_from_arrays(
        #         pair[0],
        #         pair[1],
        #         Path("/Users/dcripe/dev/ai/cv_playground/out/generated_comps", f"{idx}_a{test_labels[idx]}_p{y_pred[idx]}.jpg")
        #     )

        # find the boundary score that optimizes pos and neg
        df = pd.DataFrame(data=zip(test_labels, y_pred[:,0]), columns=['act', 'pred'])
        print(f"positive labels:\n{df[df['act'] == 1.0]['pred'].describe()}\n")
        print(f"negative labels:\n{df[df['act'] == 0.0]['pred'].describe()}\n")

        print("confusion matrix:\n")
        print(confusion_matrix(test_labels, (y_pred[:,0] < DISSIMILARITY_THRESHOLD)))

        metric = F1Score(threshold=DISSIMILARITY_THRESHOLD)
        metric.update_state(
            test_labels.reshape((test_labels.shape[0],1)),
            (y_pred[:,0] < DISSIMILARITY_THRESHOLD).reshape((test_labels.shape[0],1)),
        )
        result = metric.result()
        return result





if __name__ == '__main__':
    # att_data_dir = "/Users/dcripe/dev/ai/learning/Neural-Network-Projects-with-Python/Chapter07/att_faces"

    # digiface_data_dir = "/Users/dcripe/dev/ai/cv_playground/data/images/digiface/subjects_100000-133332_5_imgs"
    # digiface_data_dir = "/Users/dcripe/dev/ai/cv_playground/data/images/digiface/subjects_100000-100100_5_imgs"
    # digiface_data_dir = "/Users/dcripe/dev/ai/cv_playground/data/images/digiface/subjects_100000-101000_5_imgs"
    digiface_data_dir = "/Users/dcripe/dev/ai/cv_playground/data/images/digiface/subjects_0-0100_72_imgs"

    # saved_model:Path = Path("/Users/dcripe/dev/ai/cv_playground/face_detection/siamese_nn.keras")
    saved_model:Path = Path("/face_detection/models/siamese_nn_digiface.h5")
    model:Model = None
    if saved_model.exists():
        print(f"loading model from save {saved_model}")
        model = load_model(
            str(saved_model),
            custom_objects={
                'contrastive_loss': contrastive_loss,
                'euclidean_distance': euclidian_distance,
            },
            safe_mode=False,
        )

    # loader:SiameseNNDataLoader = ATTSiameseNNDataLoader(Path(att_data_dir))
    # loader:SiameseNNDataLoader = ATTSiameseNNDataLoader(Path(digiface_data_dir))
    # loader:SiameseNNDataLoader = ATTSiameseNNDataLoader(Path(digiface_data_dir_100))
    loader:SiameseNNDataLoader = ATTSiameseNNDataLoader(
        Path(digiface_data_dir),
        max_positive_pairs_per_image=2,
        max_negative_pairs_per_image=2,
    )
    loader.load(reload=True)
    net:SiameseNeuralNetTrainer = SiameseNeuralNetTrainer(model=model, loader=loader)
    net.assemble(loader.get_input_shape())
    # model = net.train(
    #     epochs=5,
    # )
    # model.save(saved_model)
    res = net.validate()
    print(f"validation f1: {res}")