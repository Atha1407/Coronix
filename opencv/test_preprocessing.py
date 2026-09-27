import cv2
from preprocessing import preprocess_image


original, enhanced = preprocess_image("sample.jpg")

cv2.imwrite("enhanced.jpg", enhanced)

print("Preprocessing completed!")
print("Original shape:", original.shape)
print("Enhanced shape:", enhanced.shape)