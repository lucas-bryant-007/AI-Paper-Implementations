import cv2
import numpy as np
from collections import deque

def down_sample_image(image):
    """Downsample any size rgb image to 84x84 grayscale"""
    image = image / 255.0
    image = np.mean(image, axis=2)
    image = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
    return image

def show_image(image):
    cv2.namedWindow("Image View Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image View Window", 300, 300)
    cv2.imshow("Image View Window", image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_row_indexed(imgs, win="stack", pad=6, scale=4):
    """
    imgs: list/tuple of 4 arrays, each (84,84) float32/float64 in [0,1]
    pad: pixels between tiles
    scale: upscaling factor for display readability
    """
    assert len(imgs) == 4
    H, W = 84, 84

    # Build 1-channel canvas (grayscale)
    canvas_h = H
    canvas_w = 4 * W + 3 * pad
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for i, img in enumerate(imgs):
        # Convert [0,1] float -> [0,255] uint8
        tile = (img * 255.0).clip(0, 255).astype(np.uint8)

        x0 = i * (W + pad)
        canvas[:, x0:x0 + W] = tile

        # Draw index label on the tile (white text on black box)
        label = str(i)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(canvas, (x0 + 2, 2), (x0 + 2 + tw + 6, 2 + th + bl + 6), 0, -1)
        cv2.putText(canvas, label, (x0 + 5, 2 + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2, cv2.LINE_AA)

    
    # Upscale for easier viewing
    if scale != 1:
        canvas = cv2.resize(canvas, (canvas_w * scale, canvas_h * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow(win, canvas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


class FrameDeque:
    def __init__(self, num_images: int):
        self.num_images = num_images
        self.buffer = deque(maxlen=num_images)

    def add(self, image):
        if len(self.buffer) < self.num_images:
            while len(self.buffer) < self.num_images:
                self.buffer.append(image)
        else:
            self.buffer.append(image)

    def return_frames(self):
        return self.buffer
