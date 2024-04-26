import cv2
import numpy as np

image = cv2.imread('etiketten/2.png', 0)  # Load the image in grayscale
image = cv2.resize(image, (256, 256))  # Resize the image if necessary

contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]  # Assuming there is only one contour

complex_contour = np.empty(contour.shape[:-1], dtype=complex)
complex_contour.real = contour[:, 0, 0].reshape(-1, 1)
complex_contour.imag = contour[:, 0, 1].reshape(-1, 1)

fourier_transform = np.fft.fft(complex_contour)
normalized_fourier_transform = np.abs(fourier_transform) / np.abs(fourier_transform[0])

reconstructed_contour = np.fft.ifft(fourier_transform)
reconstructed_contour = np.array([reconstructed_contour.real, reconstructed_contour.imag]).T
reconstructed_contour = np.expand_dims(reconstructed_contour, axis=1).astype(np.int32)

cv2.drawContours(image, [reconstructed_contour], -1, (255, 255, 255), 2)
cv2.imshow('Reconstructed Contour', image)
cv2.waitKey(0)
cv2.destroyAllWindows()