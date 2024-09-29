import cv2
import numpy as np
import random

# 1. Сделать селфи в деловом стиле
image = cv2.imread('C:\\Users\\shakh\\python_course\\data\\Gadzhiev.jpg')

# 2. Загрузить фотографию как цветную и отобразить ее на экране в отдельном окне
# средствами OpenCV
img_height, img_width = image.shape[:2]

screen_width = 1920
screen_height = 1080

scale = screen_height / img_height

new_width = int(img_width * scale)
new_height = int(img_height * scale)

resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

cv2.namedWindow("Resized Image", cv2.WINDOW_NORMAL)
cv2.imshow("Resized Image", resized_image)

cv2.imwrite('images/resized_image.jpg', resized_image)

# 3. Отразить цветную отмасштабированную фотографию слева направо.

flipped_image_np = resized_image[:, ::-1]
flipped_image_cv = cv2.flip(resized_image, 1)

combined_image = np.hstack((resized_image, flipped_image_np))

cv2.namedWindow("Original and Flipped", cv2.WINDOW_NORMAL)
cv2.imshow("Original and Flipped", combined_image)

cv2.imwrite('images/combined_image.jpg', combined_image)

# 4. Произвести обрезку цветной отмасштабированной фотографии

crop_x = int(0.15 * new_width)   # 15% от ширины
crop_y = int(0.15 * new_height)  # 15% от высоты

cropped_left = resized_image[:, crop_x:] # Обрезка слева
cropped_right = resized_image[:, :-crop_x] # Обрезка справа
cropped_top = resized_image[crop_y:, :] # Обрезка сверху
cropped_bottom = resized_image[:-crop_y, :] # Обрезка снизу

combined_lr = np.hstack((cropped_left, cropped_right))
combined_tb = np.hstack((cropped_top, cropped_bottom))

cv2.namedWindow("Left and Right Cropped", cv2.WINDOW_NORMAL)
cv2.namedWindow("Top and Bottom Cropped", cv2.WINDOW_NORMAL)

cv2.imshow("Left and Right Cropped", combined_lr)
cv2.imshow("Top and Bottom Cropped", combined_tb)

cv2.imwrite('images/left_and_right_cropped.jpg', combined_lr)
cv2.imwrite('images/top_and_bottom_cropped.jpg', combined_tb)

# 5. Произвести повороты цветной отмасштабированной фотографии
center = (new_width // 2, new_height // 2)

rotation_matrix_right = cv2.getRotationMatrix2D(center, -30, 1)
rotated_right = cv2.warpAffine(resized_image, rotation_matrix_right, (new_width, new_height))

rotation_matrix_left = cv2.getRotationMatrix2D(center, 30, 1)
rotated_left = cv2.warpAffine(resized_image, rotation_matrix_left, (new_width, new_height))

combined_rotated_images = np.hstack((rotated_left, rotated_right))

cv2.namedWindow("Rotated Images", cv2.WINDOW_NORMAL)
cv2.imshow("Rotated Images", combined_rotated_images)
cv2.imwrite('images/rotated_images.jpg', combined_rotated_images)

# 6. Провести размытие цветной отмасштабированной фотографии
blurred_image = cv2.blur(resized_image, (10, 10))

combined_blur_image = np.hstack((resized_image, blurred_image))

cv2.namedWindow("Original and Blurred", cv2.WINDOW_NORMAL)
cv2.imshow("Original and Blurred", combined_blur_image)

cv2.imwrite('images/original_and_blurred.jpg', combined_blur_image)

# 7. Произвести зашумление цветной отмасштабированной фотографии
noise = np.random.randint(0, 75, (new_height, new_width, 3), dtype='uint8')

noisy_image = cv2.add(resized_image, noise)

combined_noise_images = np.hstack((resized_image, noisy_image))

cv2.namedWindow("Original and Noisy", cv2.WINDOW_NORMAL)
cv2.imshow("Original and Noisy", combined_noise_images)
cv2.imwrite('images/original_and_noisy.jpg', combined_noise_images)

# 8. Нанести повреждения, имитирующих сгибы (или царапины) на цветной
# отмасштабированной фотографии
folds_image = resized_image
num_folds = 7

for _ in range(num_folds):
    start_point = (0, random.randint(0, new_height))  # Точка на левом краю
    end_point = (new_width - 1, random.randint(0, new_height))  # Точка на правом краю

    cv2.line(resized_image, start_point, end_point, (255, 255, 255), 1)

cv2.namedWindow("Image with Folds", cv2.WINDOW_NORMAL)
cv2.imshow("Image with Folds", folds_image)
cv2.imwrite('images/image_with_folds.jpg', folds_image)

# 9. Загрузить фотографию как изображение в оттенках серого и отмасштабировать
image = cv2.imread('C:\\Users\\shakh\\python_course\\data\\Gadzhiev.jpg', cv2.IMREAD_GRAYSCALE)

img_height, img_width = image.shape[:2]

screen_width = 1920
screen_height = 1080

scale = screen_height / img_height

new_width = int(img_width * scale)
new_height = int(img_height * scale)

resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


# 10. Протестировать работу бинаризации с адаптивным порогом
binary_3 = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 3, 2)
binary_5 = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 5, 2)
binary_7 = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 7, 2)

combined_3 = cv2.hconcat([resized_image, binary_3])
combined_5 = cv2.hconcat([resized_image, binary_5])
combined_7 = cv2.hconcat([resized_image, binary_7])

cv2.namedWindow("Adaptive Threshold 3", cv2.WINDOW_NORMAL)
cv2.imshow("Adaptive Threshold 3", combined_3)

cv2.namedWindow("Adaptive Threshold 5", cv2.WINDOW_NORMAL)
cv2.imshow("Adaptive Threshold 5", combined_5)

cv2.namedWindow("Adaptive Threshold 7", cv2.WINDOW_NORMAL)
cv2.imshow("Adaptive Threshold 7", combined_7)

cv2.imwrite('images/adaptive_threshold_3.jpg', combined_3)
cv2.imwrite('images/adaptive_threshold_5.jpg', combined_5)
cv2.imwrite('images/adaptive_threshold_7.jpg', combined_7)

cv2.waitKey(0)
cv2.destroyAllWindows()

