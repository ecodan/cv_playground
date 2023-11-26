from pathlib import Path
from typing import Dict, List

import cv2

from face_detection.face_classifier import FaceClassifier, face_ref_from_dir
from face_detection.face_detector import HaarCascadesFaceDetector, MTCNNFaceDetector, FaceDetector
import keras.backend as kbe

WHITE = (255, 255, 255)
BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

def visualize_classification(image_dir:Path, out_dir:Path, fd: FaceDetector, fc: FaceClassifier):
    '''
    Marks up all images in the test dataset with face bounding boxes, names and confidence
    :return:
    '''

    # get all faces (img + bbox) in each image
    all_faces:Dict[str,List] = fd.find_faces_in_dir(image_dir)

    for img_name in all_faces.keys():
        if len(all_faces[img_name]) == 0:
            print(f"no faces in {img_name}; skipping...")
            continue
        image = cv2.imread(str(Path(image_dir, img_name)))
        for face_img, face_bb, face_conf in all_faces[img_name]:
            x, y, w, h = face_bb
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
            name, score = fc.identify_face(face_img)
            print(f"{img_name} contains {name} (score: {score})")
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img=image,
                text=name,
                org=(x,y+h+50),
                fontFace=font,
                fontScale=2,
                color=GREEN,
                thickness=3,
                lineType=cv2.LINE_8,
            )
            cv2.putText(
                img=image,
                text=f"dist:{score:.2f}",
                org=(x,y+h+100),
                fontFace=font,
                fontScale=2,
                color=GREEN,
                thickness=3,
                lineType=cv2.LINE_8,
            )
        cv2.imwrite(str(Path(out_dir, f"annotated-{img_name}")), image)


if __name__ == '__main__':

    # run visualize classification pipeline
    # image_dir: Path = Path("/Users/dcripe/dev/ai/cv_playground/data/images/face_ref")
    image_dir: Path = Path("/Users/dcripe/dev/ai/cv_playground/data/images/test")
    # image_dir: Path = Path("/Users/dcripe/Pictures/ai/semantic/2022/20220718 - europe:africa/06 Sudtirol")
    out_dir: Path = Path("/Users/dcripe/dev/ai/cv_playground/out/generated_comps")
    ref_faces_dir: Path = Path("/Users/dcripe/dev/ai/cv_playground/data/images/face_ref")
    # detector = HaarCascadesFaceDetector()
    detector = MTCNNFaceDetector(threshold=0.96)

    # image shape required by the trained model
    shape:tuple = (112, 112)
    DISSIMILARITY_THRESHOLD:float = 0.5

    classifier = FaceClassifier(
        model_key="siamese",
        ref_faces=face_ref_from_dir(
            dir_path=ref_faces_dir,
            detector=detector,
            shape=shape,
        ),
        image_shape=shape,
        similarity_threshold=DISSIMILARITY_THRESHOLD,
    )
    visualize_classification(image_dir, out_dir, detector, classifier)
