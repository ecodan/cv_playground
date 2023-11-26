import abc
from abc import ABCMeta
from pathlib import Path
from typing import List, Dict
from enum import Enum

import keras.backend as kbe
import numpy as np
import pandas as pd
from keras import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.preprocessing import image as kpimg
from keras.src.metrics import F1Score
from tensorflow.math import confusion_matrix
from tensorflow.keras.metrics import F1Score
from tqdm import tqdm

# this is currently set manually based on eyeballing the positive P75 and negative P25 values (from Validate)
from face_detection import utils

DISSIMILARITY_THRESHOLD: float = 0.65

ALLOWED_IMAGE_TYPES = ['.jpg','.jpeg']

def euclidian_distance(vectors: List) -> float:
    v1, v2 = vectors
    return kbe.sqrt(kbe.maximum(kbe.sum(kbe.square(v1 - v2), axis=1, keepdims=True), kbe.epsilon()))


def contrastive_loss(y_true, D) -> float:
    margin = 1.0
    return kbe.mean((y_true * kbe.square(D)) + (1 - y_true) * kbe.maximum((margin - D), 0))


def accuracy(y_true, y_pred) -> float:
    return kbe.mean(kbe.equal(y_true, kbe.cast(y_pred < DISSIMILARITY_THRESHOLD, y_true.dtype)))


class TrainTestType(Enum):
    TRAIN = 1000
    TEST = 2000


class SiameseNNDataLoader(metaclass=ABCMeta):

    @abc.abstractmethod
    def load(self, train_test_split: float, reload: bool):
        raise NotImplemented()

    @abc.abstractmethod
    def get_input_shape(self) -> tuple:
        raise NotImplemented()

    @abc.abstractmethod
    def generate_pairs_and_labels(self):
        raise NotImplemented()

    @abc.abstractmethod
    def next_batch(self, train_test_type: TrainTestType, batch_size: int):
        raise NotImplemented()

    @staticmethod
    def _unique_files(pairs: List[tuple]) -> set[Path]:
        all_files: List = []
        for p1, p2 in pairs:
            all_files.extend([p1, p2])
        return set(all_files)

    @staticmethod
    def _load_images(files: set[Path]) -> Dict[str, np.ndarray]:
        images = {}
        for f in files:
            img = kpimg.load_img(str(f), color_mode='grayscale')
            img = kpimg.img_to_array(img).astype('float32') / 255
            img = img.reshape(img.shape[0], img.shape[1], 1)
            images[str(f)] = img
        return images

    @staticmethod
    def _files_to_images(pairs: np.ndarray[tuple]) -> np.ndarray[tuple]:
        images = SiameseNNDataLoader._load_images(SiameseNNDataLoader._unique_files(pairs))
        ret = []
        for pair in pairs:
            ret.append((images[str(pair[0])], images[str(pair[1])]))
        return np.array(ret)

    @abc.abstractmethod
    def has_more_batches(self, train_test_type):
        raise NotImplemented()

    @abc.abstractmethod
    def get_batch_count(self, train_test_type, batch_size):
        raise NotImplemented()

    @abc.abstractmethod
    def visual_validation(self, train_test_type, output_dir, num_samples):
        raise NotImplemented()


class ATTSiameseNNDataLoader(SiameseNNDataLoader):
    '''
    utility class to load and vend images from training datasets
    '''

    def __init__(self, root_dir: Path, max_positive_pairs_per_image: int = 5,
                 max_negative_pairs_per_image: int = 5, pair_samples_path: Path = None) -> None:
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

        self.last_train_batch_idx = 0
        self.last_test_batch_idx = 0

    def load(self, train_test_split: float = 0.75, reload: bool = True, split_by_class:bool = False) -> tuple[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ]:
        '''
        loads the image metadata into memory (images will be loaded into memory when served)
        :param train_test_split: ratio of training samples to test (1.0 = all train)
        :param reload: force reload if the data is already loaded
        :return: (tuple of training images and training labels) , (tuple of test images and test labels)
        '''
        assert train_test_split >= 0.0 and train_test_split <= 1.0
        print(f"loading data from {self.root_dir}")

        if self.img_train is None or reload:
            class_dirs: np.ndarray = np.array([x for x in self.root_dir.iterdir() if x.is_dir()])
            num_classes: int = len(class_dirs)
            if split_by_class:
                # choose the classes in train and test
                idx_samples = np.random.random_sample(num_classes)
                train_class_dirs: np.ndarray = class_dirs[idx_samples <= train_test_split]
                test_class_dirs: np.ndarray = class_dirs[idx_samples > train_test_split]

                img_train, class_train, img_test, class_test = [], [], [], []
                print(f"iterating through {num_classes} classes (train={train_test_split}...")
                for dir in tqdm(class_dirs):
                    for f in sorted([x for x in dir.iterdir() if x.is_file()]):
                        if f.suffix not in ALLOWED_IMAGE_TYPES:
                            continue
                        if dir in train_class_dirs:
                            img_train.append(f)
                            class_train.append(np.where(train_class_dirs == dir)[0][0])
                        else:
                            img_test.append(f)
                            class_test.append(np.where(test_class_dirs == dir)[0][0])
            else:
                imgs, classes = [], []
                for dir in tqdm(class_dirs):
                    for f in sorted([x for x in dir.iterdir() if x.is_file()]):
                        if f.suffix not in ALLOWED_IMAGE_TYPES:
                            continue
                        imgs.append(f)
                        classes.append(np.where(class_dirs == dir)[0][0])
                imgs = np.array(imgs)
                classes = np.array(classes)
                idx_samples = np.random.random_sample(len(imgs))
                img_train:np.ndarray = imgs[idx_samples <= train_test_split]
                img_test:np.ndarray = imgs[idx_samples > train_test_split]
                class_train:np.ndarray = classes[idx_samples <= train_test_split]
                class_test:np.ndarray = classes[idx_samples > train_test_split]

            self.img_train = np.array(img_train)
            self.class_train = np.array(class_train)
            self.img_test = np.array(img_test)
            self.class_test = np.array(class_test)

        # returns list of images (1d, each class sequentially) with matching class labels
        print(f"done loading data. train imgs={len(self.img_train)} test imgs={len(self.img_test)}")
        return (self.img_train, self.class_train), (self.img_test, self.class_test)

    def get_input_shape(self) -> tuple:
        images: Dict[str, np.ndarray] = SiameseNNDataLoader._load_images({self.img_train[0]})
        img = images[next(iter(images.keys()))]
        shape = img.shape
        return shape

    def generate_pairs_and_labels(self) -> tuple:
        '''
        creates a list of pairs and associated labels
        :return: NDArray of image pair tuples (one face each) and a matching NDArray of labels (1 = same class, 0 = different class)
        '''
        for t in [TrainTestType.TRAIN, TrainTestType.TEST]:
            print(f"generating pairs for {t.name}...")
            pairs = []
            labels = []

            # loop through each image
            if t == TrainTestType.TRAIN:
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
                    self.max_positive_pairs_per_image if self.max_positive_pairs_per_image <= len(
                        same_class_idxs) else len(same_class_idxs),
                    replace=False
                )
                for i in pos_pair_idxs:
                    pairs.append((X[idx], X[i]))
                    labels.append(1.0)

                # randomly select 5 negative pairings from other classes
                neg_pair_idxs = rng.choice(diff_class_idxs, self.max_negative_pairs_per_image, replace=False)
                for i in neg_pair_idxs:
                    pairs.append((X[idx], X[i]))
                    labels.append(0.0)

            print(f" .. done; generated {len(labels)} pairs for {t.name}")
            if t == TrainTestType.TRAIN:
                self.train_pairs = np.array(pairs)
                self.train_labels = np.array(labels)
            else:
                self.test_pairs = np.array(pairs)
                self.test_labels = np.array(labels)

        return (self.train_pairs, self.train_labels), (self.test_pairs, self.test_labels)

    def has_more_batches(self, train_test_type: TrainTestType, ) -> bool:
        assert train_test_type == TrainTestType.TRAIN or train_test_type == TrainTestType.TEST
        if train_test_type == TrainTestType.TRAIN:
            if self.last_train_batch_idx >= len(self.train_pairs):
                return False
            return True
        else:
            if self.last_test_batch_idx >= len(self.test_pairs):
                return False
            return True

    def get_batch_count(self, train_test_type: TrainTestType, batch_size=-1) -> tuple:
        assert train_test_type == TrainTestType.TRAIN or train_test_type == TrainTestType.TEST
        if batch_size == -1:
            return 0, 1
        if train_test_type == TrainTestType.TRAIN:
            completed = self.last_train_batch_idx / batch_size
            total = len(self.train_pairs) / batch_size
            return completed, total
        else:
            completed = self.last_test_batch_idx / batch_size
            total = len(self.test_pairs) / batch_size
            return completed, total

    def next_batch(self, train_test_type: TrainTestType, batch_size: int = -1, ) -> tuple:
        '''
        Serves a batch of ((img, img), label) tuples until all TRAIN images are served
        :param train_test_type: whether to draw from the TRAIN or TEST data
        :param batch_size: number if tuples to serve
        :return: ((img, img), label) tuples
        '''
        assert train_test_type == TrainTestType.TRAIN or train_test_type == TrainTestType.TEST
        if batch_size < 0:
            if train_test_type == TrainTestType.TRAIN:
                self.last_train_batch_idx = len(self.train_pairs)
                return SiameseNNDataLoader._files_to_images(self.train_pairs), self.train_labels
            else:
                self.last_test_batch_idx = len(self.test_pairs)
                return SiameseNNDataLoader._files_to_images(self.test_pairs), self.test_labels
        else:
            if train_test_type == TrainTestType.TRAIN:
                assert batch_size < len(self.train_pairs)
                if self.last_train_batch_idx + batch_size < len(self.train_pairs):
                    trn = self.train_pairs[self.last_train_batch_idx:self.last_train_batch_idx + batch_size]
                    lbl = self.train_labels[self.last_train_batch_idx:self.last_train_batch_idx + batch_size]
                    self.last_train_batch_idx += batch_size
                    return SiameseNNDataLoader._files_to_images(trn), lbl
                elif self.last_train_batch_idx == len(self.train_pairs):
                    return [], []
                else:
                    trn = self.train_pairs[self.last_train_batch_idx::]
                    lbl = self.train_labels[self.last_train_batch_idx::]
                    self.last_train_batch_idx = len(self.train_pairs)
                    return SiameseNNDataLoader._files_to_images(trn), lbl
            else:
                assert batch_size < len(self.test_pairs)
                if self.last_test_batch_idx + batch_size >= len(self.test_pairs):
                    # reset idx and shuffle lists... it's ok for validation, so we'll just keep circling around
                    p = np.random.permutation(len(self.test_pairs))
                    self.test_pairs = self.test_pairs[p]
                    self.test_labels = self.test_labels[p]
                    self.last_test_batch_idx = 0
                trn = self.test_pairs[self.last_test_batch_idx:self.last_test_batch_idx + batch_size]
                lbl = self.test_labels[self.last_test_batch_idx:self.last_test_batch_idx + batch_size]
                self.last_test_batch_idx += batch_size
                return SiameseNNDataLoader._files_to_images(trn), lbl

    def visual_validation(self, train_test_type: TrainTestType, output_dir: Path, num_samples: int = 10):
        '''
        Outputs num_sample images featuring both of a pair with the label in the filename.
        :param train_test_type: whether to draw from the TRAIN or TEST data
        :param output_dir: location of output files
        :param num_samples: number of output files
        '''
        assert train_test_type == TrainTestType.TRAIN or train_test_type == TrainTestType.TEST
        if train_test_type == TrainTestType.TRAIN:
            index = np.random.choice(self.train_pairs.shape[0], num_samples, replace=False)
            X_samples = self.train_pairs[index]
            y_samples = self.train_labels[index]
        else:
            index = np.random.choice(self.test_pairs.shape[0], num_samples, replace=False)
            X_samples = self.test_pairs[index]
            y_samples = self.test_labels[index]

        for idx, val in enumerate(X_samples):
            pair = X_samples[idx]
            label = y_samples[idx]
            utils.side_by_side_from_paths(pair[0], pair[1], Path(output_dir, f"sample_{idx}_l_{int(label)}.jpg"))


class SiameseNeuralNetTrainer():
    '''
    wraps Keras model to train a SiameseNN
    '''

    def __init__(self, loader: SiameseNNDataLoader, model: Model = None) -> None:
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

    def _create_common_model(self, input_shape: tuple):
        '''
        creates the core convolutional model that will feed the euclidean distance
        :return: Keras model
        '''
        common_model = Sequential(name='common')
        common_model.add(
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        common_model.add(MaxPooling2D())
        common_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        common_model.add(Flatten())
        common_model.add(Dense(units=128, activation='sigmoid'))
        return common_model

    def assemble(self, shape: tuple):
        '''
        creates the Siamese NN Keras model
        :param shape: the image shape to be trained on
        '''
        print(f"assempling NN with shape {shape}")
        if model is None:
            input_A = Input(shape=shape)
            input_B = Input(shape=shape)
            common_model = self._create_common_model(shape)
            output_A = common_model(input_A)
            output_B = common_model(input_B)
            distance = Lambda(euclidian_distance, output_shape=(1,))([output_A, output_B])
            self.model = Model(inputs=[input_A, input_B], outputs=distance)

    def train(self, data_load_batch_size: int = -1, epochs: int = 10, model_fit_batch_size: int = 32) -> Model:
        '''
        trains the model on pairs generate by the data loader
        :param epochs: number of training epochs
        :param model_fit_batch_size: self explanatory
        :return: trained model
        '''
        print(f"training model...")

        print(f" .. compiling model")
        self.model.compile(
            loss=contrastive_loss,
            optimizer='adam',
            metrics=[accuracy],
            # metrics=[Accuracy()],
            # metrics=F1Score(),
        )
        _, tot = self.loader.get_batch_count(TrainTestType.TRAIN, batch_size=data_load_batch_size)
        print(f" .. fitting model (est. {tot} data batches)")
        while (loader.has_more_batches(TrainTestType.TRAIN)):
            training_pairs, training_labels = self.loader.next_batch(TrainTestType.TRAIN,
                                                                     batch_size=data_load_batch_size)
            v_pairs, v_labels = self.loader.next_batch(TrainTestType.TEST, batch_size=data_load_batch_size)
            t1 = training_pairs[:, 0]
            t2 = training_pairs[:, 1]
            self.model.fit(
                [t1, t2],
                training_labels,
                batch_size=model_fit_batch_size,
                epochs=epochs,
                # validation_split=0.1,
                validation_data=([v_pairs[:, 0], v_pairs[:, 1]], v_labels),
            )
            cmplt, tot = self.loader.get_batch_count(TrainTestType.TRAIN, batch_size=data_load_batch_size)
            print(f"  .. fitted {cmplt} batches of {tot}")
        return self.model

    def validate(self) -> float:
        '''
        Utility method that outputs several training result attributes
        :return: F1Score
        '''
        print("validating model")
        test_pairs, test_labels = self.loader.next_batch(TrainTestType.TEST)
        res = self.model.evaluate(
            [test_pairs[:, 0], test_pairs[:, 1]],
            test_labels,
            batch_size=128,
            return_dict=True,
        )
        print(f"results from evaluate: {res}")

        y_pred = model.predict(
            [test_pairs[:, 0], test_pairs[:, 1]],
            batch_size=32,
        )
        # for idx, pair in enumerate(test_pairs):
        #     utils.side_by_side_from_arrays(
        #         pair[0],
        #         pair[1],
        #         Path("/Users/dcripe/dev/ai/cv_playground/out/generated_comps", f"{idx}_a{test_labels[idx]}_p{y_pred[idx]}.jpg")
        #     )

        # find the boundary score that optimizes pos and neg
        df = pd.DataFrame(data=zip(test_labels, y_pred[:, 0]), columns=['act', 'pred'])
        print(f"positive labels:\n{df[df['act'] == 1.0]['pred'].describe()}\n")
        print(f"negative labels:\n{df[df['act'] == 0.0]['pred'].describe()}\n")

        print("confusion matrix:\n")
        print(confusion_matrix(test_labels, (y_pred[:, 0] < DISSIMILARITY_THRESHOLD)))

        metric = F1Score(threshold=DISSIMILARITY_THRESHOLD)
        metric.update_state(
            test_labels.reshape((test_labels.shape[0], 1)),
            (y_pred[:, 0] < DISSIMILARITY_THRESHOLD).reshape((test_labels.shape[0], 1)),
        )
        result = metric.result()
        return result


if __name__ == '__main__':
    # att_data_dir = "/Users/dcripe/dev/ai/learning/Neural-Network-Projects-with-Python/Chapter07/att_faces"

    # digiface_data_dir = "/Users/dcripe/Pictures/ai/training/digiface/subjects_100000-133332_5_imgs"
    # digiface_data_dir = "/Users/dcripe/Pictures/ai/training/digiface/subjects_100000-100100_5_imgs"
    # digiface_data_dir = "/Users/dcripe/Pictures/ai/training/digiface/subjects_100000-101000_5_imgs"
    # digiface_data_dir = "/Users/dcripe/Pictures/ai/training/digiface/subjects_0-0100_72_imgs"
    # digiface_data_dir = "/Users/dcripe/Pictures/ai/training/digiface/subjects_0-1999_72_imgs"

    cripe_data_dir = "/Users/dcripe/Pictures/ai/training/cripe/faces_112x112"

    # saved_model:Path = Path("/Users/dcripe/dev/ai/cv_playground/face_detection/siamese_nn.keras")
    saved_model: Path = Path(
        f"/Users/dcripe/dev/ai/cv_playground/face_detection/models/siamese_nn_cripe_01.h5")
    model: Model = None
    # if saved_model.exists():
    #     print(f"loading model from save {saved_model}")
    #     model = load_model(
    #         str(saved_model),
    #         custom_objects={
    #             'contrastive_loss': contrastive_loss,
    #             'euclidean_distance': euclidian_distance,
    #         },
    #         safe_mode=False,
    #     )

    # loader:SiameseNNDataLoader = ATTSiameseNNDataLoader(Path(att_data_dir))
    # loader:SiameseNNDataLoader = ATTSiameseNNDataLoader(Path(digiface_data_dir))
    # loader:SiameseNNDataLoader = ATTSiameseNNDataLoader(Path(digiface_data_dir_100))
    loader: SiameseNNDataLoader = ATTSiameseNNDataLoader(
        # Path(digiface_data_dir),
        Path(cripe_data_dir),
        max_positive_pairs_per_image=25,
        max_negative_pairs_per_image=25,
    )
    loader.load(reload=True)
    loader.generate_pairs_and_labels()
    loader.visual_validation(
        train_test_type=TrainTestType.TRAIN,
        output_dir=Path("/Users/dcripe/dev/ai/cv_playground/out/generated_comps"),
        num_samples=50
    )

    net: SiameseNeuralNetTrainer = SiameseNeuralNetTrainer(model=model, loader=loader)
    net.assemble(loader.get_input_shape())
    model = net.train(
        # data_load_batch_size=10000,
        data_load_batch_size=-1,
        epochs=4,
    )
    model.save(saved_model)
    res = net.validate()
    print(f"validation f1: {res}")
