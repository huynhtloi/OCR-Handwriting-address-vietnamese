import cv2
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
# import matplotlib.pyplot as plt

def preprocessing_img(bounding_box_img):
    TIME_STEPS = 240  # Define your desired time steps here

    # Convert image to grayscale
    img_gray = cv2.cvtColor(bounding_box_img, cv2.COLOR_BGR2GRAY)

    height, width = img_gray.shape

    # Resize image to have height = 118 while maintaining the aspect ratio
    new_height = 118
    new_width = int((width / height) * new_height)
    img_resized = cv2.resize(img_gray, (new_width, new_height))

    # Padding image using median
    img_padded = np.pad(img_resized, ((0, 0), (0, max(0, 2167 - new_width))), 'median')

    # Blur it
    img_blurred = cv2.GaussianBlur(img_padded, (5, 5), 0)

    # Threshold the image using adaptive threshold - binary images
    img_thresholded = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Add channel dimension
    img_expanded = np.expand_dims(img_thresholded, axis=2)

    # Normalizes the pixel value of the image to the range from 0 to 1.
    img_normalized = img_expanded / 255.

    print(img_normalized.shape)

    return img_normalized

def padding_image(image, target_width):
    current_height, current_width, channels = image.shape  # Lấy số kênh của ảnh gốc
    padding_width = target_width - current_width
    if padding_width > 0:
        # Tạo ảnh đệm có kênh giống với ảnh gốc
        padding = np.zeros((current_height, padding_width, channels), dtype=np.uint8)
        image = np.concatenate((image, padding), axis=1)
    return image

def bounding_box(image):
    # Khởi tạo bộ nhận dạng văn bản
    reader = easyocr.Reader(lang_list=['vi'])

    # Nhận dạng văn bản trong ảnh
    results = reader.readtext(image)

    # Sắp xếp các bounding box theo tọa độ Y
    results.sort(key=lambda x: x[0][0][1])

    # Xác định khoảng cách ngang tối đa để gộp bounding box
    max_horizontal_gap = -40  # Tùy chỉnh theo kích thước của ảnh

    # Tách riêng các dòng văn bản
    text_lines = []

    test_img = []

    current_line = [results[0]]
    for i in range(1, len(results)):
        if results[i][0][0][1] - current_line[-1][0][2][1] <= max_horizontal_gap:
            current_line.append(results[i])
        else:
            text_lines.append(current_line)
            current_line = [results[i]]
            
    text_lines.append(current_line)
    
    # Vẽ bounding box cho từng dòng văn bản riêng biệt
    for line in text_lines:
        top_left = (min(bbox[0][0][0] for bbox in line), min(bbox[0][0][1] for bbox in line))
        bottom_right = (max(bbox[0][2][0] for bbox in line), max(bbox[0][2][1] for bbox in line))

        # Extract bounding box region
        x, y = int(top_left[0]), max(0, int(top_left[1]))
        w, h = int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])

        # Phóng to chiều cao lên 118 và tính toán chiều rộng mới
        new_height = 118
        new_width = int((w / h) * new_height)
        bounding_box_img = cv2.resize(image[y:y+h, x:x+w], (new_width, new_height))
        if new_width >= 250:
            # Padding phía bên phải để đảm bảo chiều dài là 2167
            padding_width = 2167 - new_width
            if padding_width > 0:
                bounding_box_img = padding_image(bounding_box_img, 2167)

            # Preprocess the bounding box image
            preprocessed_img = preprocessing_img(bounding_box_img)

            test_img.append(preprocessed_img)

            # Now you can use the preprocessed_img for further processing

            # Display the preprocessed image (for demonstration purposes)
            # plt.imshow(preprocessed_img.squeeze(), cmap='gray')
            # plt.show()

    return test_img