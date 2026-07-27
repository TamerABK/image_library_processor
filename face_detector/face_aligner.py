from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class FaceAligner:
    """
    Aligns a face to the canonical ArcFace/SFace 112x112 template.
    """

    output_size: tuple[int, int] = (112, 112)

    REFERENCE_LANDMARKS = np.array(
        [
            [38.2946, 51.6963],   # left eye
            [73.5318, 51.5014],   # right eye
            [56.0252, 71.7366],   # nose
            [41.5493, 92.3655],   # left mouth
            [70.7299, 92.2041],   # right mouth
        ],
        dtype=np.float32,
    )

    def align(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
    ) -> np.ndarray:
        """
        Align a face using five facial landmarks.

        Parameters
        ----------
        image
            Original BGR image.

        landmarks
            np.ndarray of shape (5,2)
            [[left_eye],
             [right_eye],
             [nose],
             [left_mouth],
             [right_mouth]]

        Returns
        -------
        np.ndarray
            112x112 aligned BGR face.
        """

        landmarks = np.asarray(
            landmarks,
            dtype=np.float32,
        )

        if landmarks.shape != (5, 2):
            raise ValueError(
                f"Expected landmarks of shape (5,2), got {landmarks.shape}"
            )

        transform, _ = cv2.estimateAffinePartial2D(
            landmarks,
            self.REFERENCE_LANDMARKS,
            method=cv2.LMEDS,
        )

        if transform is None:
            raise RuntimeError("Failed to estimate face alignment transform.")

        return cv2.warpAffine(
            image,
            transform,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )