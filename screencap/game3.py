import cv2

from skimage.metrics import structural_similarity as compare_ssim


def main():
    root_dir = '/Users/zhouzhenliang/Desktop/zlz/'
    x1, y1 = 0, 0
    img1 = cv2.imread(f'{root_dir}/z_out_{x1}-{y1}.jpg')
    # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    for i in range(0, 10):
        for j in range(0, 14):
            img2 = cv2.imread(f'{root_dir}/z_out_{i}-{j}.jpg')
            # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            score, _ = compare_ssim(img1, img2, win_size=41, full=True)
            if score > 0.9:
                print(f'({x1},{y1}) vs. ({i},{j}) = {score}')


if __name__ == '__main__':
    main()
