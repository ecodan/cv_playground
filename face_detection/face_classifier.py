from pathlib import Path
from typing import List, Dict
from keras import Model
import cv2
import numpy as np
import json

from face_detection import utils
from face_detection.face_detector import HaarCascadesFaceDetector, FaceDetector, MTCNNFaceDetector
from face_detection.siamese_nn import SiameseNeuralNetTrainer, contrastive_loss, euclidian_distance
from keras.models import load_model
import keras.backend as kbe


ALLOWED_IMAGE_TYPES = ['.jpg','.jpeg']

# this is currently set manually based on eyeballing the positive P75 and negative P25 values (from Validate)
DISSIMILARITY_THRESHOLD:float = 0.65

def load_siamese_model() -> Model:
    # saved_model:Path = Path("/Users/dcripe/dev/ai/cv_playground/face_detection/models/siamese_nn_digiface.h5")
    # saved_model:Path = Path("/Users/dcripe/dev/ai/cv_playground/face_detection/models/siamese_nn_digiface_subjects_100000-101000_5_imgs.h5")
    saved_model:Path = Path("/Users/dcripe/dev/ai/cv_playground/face_detection/models/siamese_nn_cripe_01.h5")
    print(f"loading siamese model from {saved_model}")
    model = None
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
    return model

def load_facenet_model() -> Model:
    saved_model = '/Users/dcripe/dev/ai/cv_playground/face_detection/models/facenet_keras.h5'
    print(f"loading facenet model from {saved_model}")
    model = load_model(saved_model)
    return model

MODEL_LOADERS = {
    'siamese': load_siamese_model,
    'facenet': load_facenet_model,
}


def dump_diagnostics(y_act:List, y_pred:List):
    '''
    utility funtion to display classification metrics
    :param y_act: List of actuals represented as 1s and 0s
    :param y_pred: List of predictions represented as 1s and 0s
    :return: accuracy, precision, recall, confusion_matrix:List
    '''
    assert len(y_act) == len(y_pred)
    tp, tn, fp, fn = 0,0,0,0
    for idx in range(len(y_act)):
        if y_act[idx] == 1 and y_pred[idx] == 1:
            tp += 1
        elif y_act[idx] == 1 and y_pred[idx] == 0:
            fn += 1
        elif y_act[idx] == 0 and y_pred[idx] == 1:
            fp += 1
        else:
            tn += 1

    cf = [[tp, fn],[fp, tn]]
    print(f"confusion matrix:\n{cf}")

    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = float(tp + tn) / len(y_act) if len(y_act) > 0 else 0
    print(f"accuracy: {accuracy} | precision: {precision} | recall: {recall}")

    return accuracy, precision, recall, cf


def face_ref_from_dir(dir_path:Path, detector:FaceDetector, shape:tuple) -> List[tuple]:
    assert dir_path.is_dir()
    ref = []
    for f in dir_path.iterdir():
        if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_TYPES:
            faces = detector.find_faces_in_image(f)
            if len(faces) == 0:
                print(f"WARNING no faces found in {f}")
            else:
                ref.append((f.stem, FaceClassifier._resize_image(faces[0][0], shape))) # use the first extracted face
    return ref



class FaceClassifier:
    '''
    Finds all face bounding boxes in a given image and then uses a SiameseNN model to match them with one of a set of reference faces.

    Reference faces are loaded from a directory containing one picture per face with a filename [name].jpg using the classmethod below.

    '''

    def __init__(self, model_key:str, ref_faces:List[tuple], image_shape:tuple[int, int], similarity_threshold:float = 0.5) -> None:
        '''
        Initializes an instance with...
        :param model_key: dict key in MODEL_LOADERS referencing a loading function
        :param ref_faces: a list of tuples (name:str, image:np.array) representing each reference face. Recommended to use the face_ref_from_dir() method below.
        :param image_shape: the h and w required by the model
        :param similarity_threshold: predictions below this will be considered positive matches.
        '''
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.face_ref:List[tuple] = ref_faces
        self.image_shape = image_shape
        self.detector:FaceDetector = HaarCascadesFaceDetector()

        self.model:Model = MODEL_LOADERS[model_key]()

    @classmethod
    def _resize_image(cls, img:np.ndarray, shape:tuple) -> np.ndarray:
        '''
        Resized the images to the size required by the model.
        :param img:
        :return:
        '''
        if len(img.shape) == 3: # has a color channel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32')/255
        img = cv2.resize(img, (shape[0], shape[1]))
        img = img.reshape(1, img.shape[0], img.shape[1], 1)
        return img

    def _find_best_match(self, img:np.ndarray) -> tuple:
        assert img is not None
        scores = []
        for ref_face in self.face_ref:
            res = self.model.predict([
                [
                    ref_face[1],
                    FaceClassifier._resize_image(img, self.image_shape)
                ]
            ])
            scores.append(res[0][0])
        z = zip([x[0] for x in self.face_ref], scores)
        for s in z:
            print(f"  {s[0]}: {s[1]}")
        scores = np.array(scores)
        min_idx = scores.argmin()
        if scores[min_idx] <= self.similarity_threshold:
            return self.face_ref[min_idx][0], self.face_ref[min_idx][1], scores[min_idx]
        else:
            return None, None, None

    def identify_faces_in_image(self, image_path:Path) -> List:
        assert image_path.is_file()
        if image_path.suffix.lower() not in ALLOWED_IMAGE_TYPES:
            print(f"{image_path} invalid file type")
            return []
        faces = self.detector.find_faces_in_image(image_path)
        ret = []
        for face in faces:
            name, img, score = self._find_best_match(face[0])
            if name is not None:
                # print(f"{image_path} contains {name}; score={score}")
                # utils.side_by_side_from_arrays(face[0], img, Path(f'/Users/dcripe/dev/ai/cv_playground/out/generated_comps/{name}_{score}.jpg'))
                ret.append((name, score))
        return ret

    def identify_face(self, img:np.ndarray) -> tuple[str, float]:
        name, img, score = self._find_best_match(img)
        if name is not None:
            # print(f"face image contains {name}; score={score}")
            # utils.side_by_side_from_arrays(face[0], img, Path(f'/Users/dcripe/dev/ai/cv_playground/out/generated_comps/{name}_{score}.jpg'))
            return name, score
        return None, 0.0


if __name__ == '__main__':

    # image shape required by the trained model
    shape:tuple = (112, 112)

    # detector = HaarCascadesFaceDetector()
    detector = MTCNNFaceDetector()

    # image_dir: Path = Path("/Users/dcripe/dev/ai/cv_playground/data/images/face_ref")
    image_dir: Path = Path("/Users/dcripe/dev/ai/cv_playground/data/images/test")
    # image_dir: Path = Path("/Users/dcripe/Pictures/ai/semantic/2022/20220718 - europe:africa/06 Sudtirol")
    out_dir: Path = Path("/Users/dcripe/dev/ai/cv_playground/out/generated_comps")
    ref_faces_dir: Path = Path("/Users/dcripe/dev/ai/cv_playground/data/images/face_ref")


    fc = FaceClassifier(
        # 'facenet',
        'siamese',
        face_ref_from_dir(
            dir_path=ref_faces_dir,
            detector=detector,
            shape=shape,
        ),
        image_shape=shape,
        similarity_threshold=DISSIMILARITY_THRESHOLD,
    )

    ground_truth_faces:Dict = None
    with open('/Users/dcripe/dev/ai/cv_playground/data/images/test/ground_truth_names.json') as f:
        ground_truth_faces = json.load(f)

    all_names = []
    for v in ground_truth_faces.values():
        all_names.extend(v)
    all_names = set(all_names)

    data = {name:{'act':[],'pred':[]} for name in all_names}
    test_images_dir = image_dir
    for f in test_images_dir.iterdir():
        if f.suffix not in ALLOWED_IMAGE_TYPES:
            continue
        faces = [x[0] for x in fc.identify_faces_in_image(f)]
        gt = ground_truth_faces[f.name]
        for name in data.keys():
            data[name]['act'].append(1 if name in gt else 0)
            data[name]['pred'].append(1 if name in faces else 0)

    for name in data.keys():
        print(f"metrics for {name}:")
        dump_diagnostics(data[name]['act'], data[name]['pred'])
