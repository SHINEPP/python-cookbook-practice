import cv2

channels = [0, 1, 2]
hist_size = [8, 8, 8]
ranges = [0, 256, 0, 256, 0, 256]


def calc_hist(img):
    return cv2.calcHist([img], channels, None, hist_size, ranges)


def compare_hist(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def main():
    x_count = 10
    y_count = 14
    root_dir = '/Users/zhouzhenliang/Desktop/zlz/'
    img = cv2.imread(f'{root_dir}/src_1.jpg')
    print(f'srcImage = {img.shape}')
    h, w = img.shape[:2]
    cropped = img[524:h - 554, 40:w - 40]
    print(f'cropped = {cropped.shape}')
    cv2.imwrite(f'{root_dir}/src_2.jpg', cropped)
    h, w = cropped.shape[:2]
    x_size, y_size = int(w / x_count), int(h / y_count)
    padding = 10
    results = []
    for i in range(0, x_count):
        for j in range(0, y_count):
            x, y = i * x_size, j * y_size
            img = cropped[y + padding:y + y_size - padding, x + padding:x + x_size - padding]
            path = f'{root_dir}/z_out_{i}-{j}.jpg'
            cv2.imwrite(path, img)
            results.append((i, j, calc_hist(img)))

    # 相同图片分组
    groups_list = []
    for result in results:
        have_group = False
        for groups in groups_list:
            if len(groups) > 0:
                score = compare_hist(result[2], groups[0][2])
                if score > 0.999:
                    groups.append(result)
                    have_group = True
        if not have_group:
            groups_list.append([result])

    for groups in groups_list:
        for vs in groups:
            print(f'{vs[0]},{vs[1]}  ', end='')
        print()


if __name__ == '__main__':
    main()
