from pathlib import Path
from unittest import TestCase

import numpy as np

from face_detection import siamese_nn


class TestATTSiameseNNDataLoader(TestCase):
    def test_load(self):
        print("test_load")
        loader = siamese_nn.ATTSiameseNNDataLoader(Path("/Users/dcripe/dev/ai/learning/Neural-Network-Projects-with-Python/Chapter07/att_faces"))
        (X_train, Y_train), (X_test, Y_test) = loader.load()
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(Y_train)
        self.assertIsNotNone(Y_test)
        self.assertEqual(300, len(X_train)) # 40 classes * .75 * 10 images per class
        self.assertEqual(300, len(Y_train)) # 40 classes * .75 * 10 images per class
        self.assertEqual(100, len(X_test)) # 40 classes * .25 * 10 images per class
        self.assertEqual(100, len(Y_test)) # 40 classes * .25 * 10 images per class
        self.assertEqual(0, sorted(Y_train)[0])
        self.assertEqual(29, sorted(Y_train)[-1])
        self.assertEqual(0, sorted(Y_test)[0])
        self.assertEqual(9, sorted(Y_test)[-1])

    def test_generate_pairs(self):
        print("test_generate_pairs")
        loader = siamese_nn.ATTSiameseNNDataLoader(Path("/Users/dcripe/dev/ai/learning/Neural-Network-Projects-with-Python/Chapter07/att_faces"))
        (X_train, Y_train), (X_test, Y_test) = loader.load()
        self.assertIsNotNone(X_train)
        train_pairs, train_labels = loader.generate_pairs_and_labels()
        self.assertIsInstance(train_pairs, np.ndarray)
        self.assertIsInstance(train_labels, np.ndarray)
        self.assertEqual(300*5*2, len(train_pairs))
        self.assertEqual(300*5*2, len(train_labels))
        # test_pairs, test_labels = loader.generate_pairs_and_labels(use_train=False, max_positive_pairs_per_image=3, max_negative_pairs_per_image=4)
        # self.assertEqual(100*(3+4), len(test_pairs))
        # self.assertEqual(100*(3+4), len(test_labels))
