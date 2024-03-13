import cv2


def main():
    root_dir = '/Users/zhouzhenliang/Desktop/zlz/'
    x1, y1 = 0, 7

    channels = [0, 1, 2]
    hist_size = [8, 8, 8]
    ranges = [0, 256, 0, 256, 0, 256]
    img1 = cv2.imread(f'{root_dir}/z_out_{x1}-{y1}.jpg')
    hist1 = cv2.calcHist([img1], channels, None, hist_size, ranges)

    for i in range(0, 10):
        for j in range(0, 14):
            img2 = cv2.imread(f'{root_dir}/z_out_{i}-{j}.jpg')
            hist2 = cv2.calcHist([img2], channels, None, hist_size, ranges)
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            if score > 0.9999:
                print(f'({x1},{y1}) vs. ({i},{j}) = {score}')


if __name__ == '__main__':
    main()
