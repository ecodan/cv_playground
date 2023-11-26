import math
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN


class FaceDetector(metaclass=ABCMeta):

    @abstractmethod
    def find_faces_in_image(self, image_path):
        '''
        Extracts portions of image contained by bounding boxes plus the bounding box coordinates for all faces in an image.
        :param image_path: pathlib Path to the image to process
        :returns List of ( np.ndarry, List ) for each face. The np.ndarray is the image data in CV2 format; the List is x,y,h,w coordinates.
        '''
        pass

    def find_faces_in_dir(self, dir: Path, limit:int = -1) -> Dict:
        '''
        convenience method that process all images in a directory
        :param dir: pathlib Path to dir to process
        :param debug: if True, will output a version of the image to the same dir annotated with bounding boxes
        :return: Dict of file_name:[list of faces(img, bbox)]
        '''
        print(f"finding faces in director {dir}")
        images = {}
        ct = 0
        for f in dir.iterdir():
            if f.suffix in ['.jpg', '.jpeg']:
                ct += 1
                if limit > 0 and ct > limit:
                    break
                faces = self.find_faces_in_image(f)
                images[f.name] = faces
        print(f"processed all images in {dir}; returning {len(images)} images of face data.")
        return images

    def output_annoted_images(self, img_root: Path, images:Dict[str, List], out_dir:Path):
        assert img_root.exists() and img_root.is_dir()
        assert out_dir.exists() and out_dir.is_dir()

        for img_name, faces in images.items():
            if len(faces) > 0:
                # add bounding boxes and append to output
                image = cv2.imread(str(Path(img_root, img_name)))
                for face in faces:
                    x, y, w, h = face[1]
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.imwrite(str(Path(out_dir, f"annotated-{img_name}")), image)

    def output_face_images(self, img_root: Path, images:Dict[str, List], out_dir:Path, shape:tuple = None):
        assert img_root.exists() and img_root.is_dir()
        assert out_dir.exists() and out_dir.is_dir()

        for img_name, faces in images.items():
            if len(faces) > 0:
                if shape is None:
                    for idx, face in enumerate(faces):
                        img_path = Path(img_root, img_name)
                        cv2.imwrite(
                            str(Path(out_dir, f"{img_path.stem}-f{idx}-c{face[2]:0.2f}.jpg")),
                            face[0]
                        )
                else:
                    image = cv2.imread(str(Path(img_root, img_name)))
                    for idx, face in enumerate(faces):
                        x, y, w, h = face[1]
                        tgt_h, tgt_w = shape
                        if w/h > tgt_w/tgt_h:
                            # need to outfill height to get to ratio
                            new_h:int = int(w * tgt_h / tgt_w)
                            adj_h = int((new_h-h) / 2)
                            if (y - adj_h < 0) or (y + h + adj_h > image.shape[0]):
                                print(f"adjusted crop exceeds image height boundary for {img_name}")
                                continue
                            y = y - adj_h
                            h = new_h
                        else:
                            # need to outfill width to get to ratio
                            new_w:int = int(tgt_w * h / tgt_h)
                            adj_w:int = int((new_w-w) / 2)
                            if (x - adj_w < 0) or (x + w + adj_w > image.shape[1]):
                                print(f"adjusted crop exceeds image width boundary for {img_name}")
                                continue
                            x = x - adj_w
                            w = new_w
                        assert abs(w/h - tgt_w/tgt_h) < 0.1
                        crop = image[y:y+h, x:x+w]
                        crop = cv2.resize(crop, shape)
                        img_path = Path(img_root, img_name)
                        cv2.imwrite(
                            str(Path(out_dir, f"{img_path.stem}-f{idx}-c{face[2]:0.2f}.jpg")),
                            crop
                        )




class HaarCascadesFaceDetector(FaceDetector):

    def __init__(self) -> None:
        super().__init__()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def find_faces_in_image(self, image_path: Path) -> List:
        '''
        Extracts portions of image contained by bounding boxes plus the bounding box coordinates for all faces in an image.
        :param image_path: pathlib Path to the image to process
        :returns List of ( np.ndarry, List ) for each face. The np.ndarray is the image data in CV2 format; the List is x,y,h,w coordinates.
        '''
        print(f"finding faces in image {image_path}")
        image = cv2.imread(str(image_path))
        greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.detector.detectMultiScale(
            greyscale,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        faces = []
        for (x, y, w, h) in detected_faces:
            face_box_image = greyscale[y:y + h, x:x + w]
            # face_box_image = face_box_image.reshape(face_box_image.shape[0], face_box_image.shape[1], 1)
            face_box_coords = [x, y, w, h]
            faces.append((face_box_image, face_box_coords, 0.00))
        print(f" .. found {len(faces)} faces")
        return faces


class MTCNNFaceDetector(FaceDetector):

    def __init__(self, threshold:float = 0.92) -> None:
        super().__init__()
        self.detector = MTCNN()
        self.threshold = threshold

    def find_faces_in_image(self, image_path: Path) -> List:
        '''
        Extracts portions of image contained by bounding boxes plus the bounding box coordinates for all faces in an image.
        :param image_path: pathlib Path to the image to process
        :returns List of ( np.ndarry, List ) for each face. The np.ndarray is the image data in CV2 format; the List is x,y,h,w coordinates.
        '''
        print(f"finding faces in image {image_path}")
        # load image from file
        # image:Image = Image.open(image_path)
        image:np.ndarray = cv2.imread(str(image_path))

        pixels = np.asarray(image)
        results = self.detector.detect_faces(pixels)
        faces = []
        for face in results:
            confidence = face['confidence']
            if confidence < self.threshold:
                continue
            # extract the bounding box from the first face
            x, y, w, h = face['box']
            # deal with negative pixel index
            x, y = abs(x), abs(y)
            x2, y2 = x + w, y + h
            # extract the face
            face_box_image = pixels[y:y2, x:x2]
            face_box_coords = [x, y, w, h]
            faces.append((face_box_image, face_box_coords, confidence))
        print(f" .. found {len(faces)} faces")
        return faces


if __name__ == '__main__':
    # image_dir: Path = Path("/Users/dcripe/dev/ai/cv_playground/data/images/face_ref")

    # image_dir: Path = Path("/Users/dcripe/Pictures/ai/semantic/2022/20220718 - europe:africa/06 Sudtirol")
    image_dir: Path = Path("/Users/dcripe/Pictures/ai/training/cripe/orig/carter")
    out_dir: Path = Path("/Users/dcripe/dev/ai/cv_playground/out/generated_comps")
    # fd = HaarCascadesFaceDetector()
    fd = MTCNNFaceDetector()
    images = fd.find_faces_in_dir(
        image_dir,
        # limit=5,
    )
    # fd.output_annoted_images(img_root=image_dir, images=images, out_dir=out_dir)
    fd.output_face_images(img_root=image_dir, images=images, out_dir=out_dir, shape=(112, 112))
