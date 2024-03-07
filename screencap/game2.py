from collections import defaultdict

import cv2
import numpy as np


def p_hash(img):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (32, 32))  # , interpolation=cv2.INTER_CUBIC

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash_v = []
    average = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > average:
                hash_v.append(1)
            else:
                hash_v.append(0)
    return hash_v


def cmp_hash(hash1, hash2):
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return 0
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return 1.0 - n / len(hash1)


def main():
    x_count = 10
    y_count = 14
    root_dir = '/Users/zhouzhenliang/Desktop/zlz/'
    img = cv2.imread(f'{root_dir}/src_1.jpg')
    print(f'srcImage = {img.shape}')
    h, w = img.shape[:2]
    cropped = img[524:h - 554, 40:w - 40]
    # cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    print(f'cropped = {cropped.shape}')
    cv2.imwrite(f'{root_dir}/src_2.jpg', cropped)
    h, w = cropped.shape[:2]
    x_size, y_size = int(w / x_count), int(h / y_count)
    padding = 10
    results = []
    for i in range(0, x_count):
        for j in range(0, y_count):
            x, y = i * x_size, j * y_size
            b_img = cropped[y + padding:y + y_size - padding, x + padding:x + x_size - padding]
            values = p_hash(b_img)
            path = f'{root_dir}/z_out_{i}-{j}.jpg'
            # print(f'{i}-{j} -> {values}')
            # cv2.imwrite(path, b_img)
            results.append((i, j, values))

    # 相同图片分组
    groups = []
    for values in results:
        processed = False
        for group in groups:
            if len(group) > 0:
                v = group[0]
                ratio = cmp_hash(values[2], v[2])
                if ratio > 0.9:
                    group.append(values)
                    processed = True
        if not processed:
            groups.append([values])

    for group in groups:
        for vs in group:
            print(f'{vs[0]},{vs[1]} ', end='')
        print()


if __name__ == '__main__':
    main()
