import cv2

if __name__ == '__main__':
    x_count = 10
    y_count = 14
    root_dir = '/Users/zhouzhenliang/Desktop/zlz/'
    img = cv2.imread(f'{root_dir}/zlz_1.jpg')
    print(f'srcImage = {img.shape}')
    h, w = img.shape[:2]
    cropped = img[524:h - 554, 40:w - 40]
    print(f'cropped = {cropped.shape}')
    cv2.imwrite(f'{root_dir}/zlz_2.jpg', cropped)
    h, w = cropped.shape[:2]
    x_size, y_size = int(w / x_count), int(h / y_count)
    for i in range(0, x_count):
        for j in range(0, y_count):
            x, y = i * x_size, j * y_size
            b_img = cropped[y:y + y_size, x:x + x_size]
            cv2.imwrite(f'{root_dir}/zlz_2_{i}-{j}.jpg', b_img)
