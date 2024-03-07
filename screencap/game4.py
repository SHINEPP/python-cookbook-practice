import cv2
import numpy as np

channels = [0, 1, 2]
hist_size = [8, 8, 8]
ranges = [0, 256, 0, 256, 0, 256]

x_count = 10
y_count = 14


def calc_hist(img):
    """
    直方图
    :param img:
    :return:
    """
    return cv2.calcHist([img], channels, None, hist_size, ranges)


def compare_hist(hist1, hist2):
    """
    直方图对比，[0,1] 值越大越相似
    :param hist1:
    :param hist2:
    :return:
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def calc_hash(img):
    """
    感知哈希算法
    :param img:
    :return:
    """
    # 缩放32*32
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)

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


def compare_hash(hash1, hash2):
    """
    Hash值对比, [0,1] 值越大越相似
    算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    :param hash1:
    :param hash2:
    :return:
    """
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


def find_blocks():
    """
    利用图像识别技术输出命名方块矩阵
    :return:
    """
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
            results.append((i, j, calc_hist(img), calc_hash(img)))

    # 相同图片分组
    groups_list = []
    for result in results:
        to_group = False
        for groups in groups_list:
            if len(groups) == 0:
                continue
            scope_scope = compare_hist(result[2], groups[0][2])
            if scope_scope < 0.99:
                continue
            scope_hash = compare_hash(result[3], groups[0][3])
            if scope_hash < 0.8:
                continue
            groups.append(result)
            to_group = True
        if not to_group:
            groups_list.append([result])

    blocks = []
    for x in range(0, x_count):
        blocks.append(['' for j in range(0, y_count)])

    group_index = 0
    for groups in groups_list:
        group_index += 1
        name = f'B{"%02d" % group_index}'
        print(f'{name}: ', end='')
        for result in groups:
            print(f'({result[0]},{result[1]}) ', end='')
            blocks[result[0]][result[1]] = name
        print()

    print()
    print('-------------------------------------------------------------')
    for y in range(0, y_count):
        print('|', end='')
        for x in range(0, x_count):
            name = blocks[x][y]
            print(f' {name} |', end='')
        print()
        print('-------------------------------------------------------------')

    return blocks


if __name__ == '__main__':
    find_blocks()
