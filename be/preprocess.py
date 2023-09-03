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

  # in this dataset, we don't need to do any resize at all here.
  img_resized = cv2.resize(img_gray, (int(118/height*width), 118))

  height, width = img_resized.shape

  # Padding image using median
  img_padded = np.pad(img_resized, ((0, 0), (0, max(0, 2167 - width))), 'median')

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

    # cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), (0, 255, 0), 2)

    # Extract bounding box region
    x, y = int(top_left[0]), max(0, int(top_left[1]))
    w, h = int(bottom_right[0] - top_left[0]), int(bottom_right[1] - top_left[1])
    # print("x:", x, "y:", y, "w:", w, "h:" , h)
    if y + h <= image.shape[0] and x + w <= image.shape[1]:
        bounding_box_img = cv2.resize(image[y:y+h, x:x+w], (2167, 118))
    else:
        print("Invalid cropping dimensions")

    # plt.imshow(bounding_box_img.squeeze(), cmap='gray')
    # plt.show()
    # print(bounding_box_img)
    # Preprocess the bounding box image
    preprocessed_img = preprocessing_img(bounding_box_img)

    test_img.append(preprocessed_img)

    # Now you can use the preprocessed_img for further processing

    # Display the preprocessed image (for demonstration purposes)
    # plt.imshow(preprocessed_img.squeeze(), cmap='gray')
    # plt.show()
    
  return test_img
