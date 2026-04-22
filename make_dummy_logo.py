"""Create a simple 'DUMMY' text logo PNG with transparency."""
import cv2
import numpy as np

w, h = 800, 300
img = np.zeros((h, w, 4), dtype=np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
text = "DUMMY"
scale = 4.0
thickness = 12
(tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
tx = (w - tw) // 2
ty = (h + th) // 2

img[:, :, :3] = 255
cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255, 255), thickness, cv2.LINE_AA)

alpha = np.zeros((h, w), dtype=np.uint8)
cv2.putText(alpha, text, (tx, ty), font, scale, 255, thickness, cv2.LINE_AA)
img[:, :, 3] = alpha

cv2.imwrite("dummy_logo.png", img)
print(f"Saved dummy_logo.png ({w}x{h})")
