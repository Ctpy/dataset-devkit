from abc import ABC
import numpy as np


class LabelObject(ABC):

    def __init__(self, pos_x, pos_y, pos_z, length, width, height, label_class):
        self.x = pos_x
        self.y = pos_y
        self.z = pos_z
        self.length = length
        self.width = width
        self.height = height
        self.label_class = label_class

    def transform(self, transformation_matrix: np.ndarray) -> None:
        # TODO
        pass

    def translate(self, translation_matrix: np.ndarray) -> None:
        # TODO
        pass

    def rotation(self, rotation_matrix: np.ndarray) -> None:
        # TODO
        pass
