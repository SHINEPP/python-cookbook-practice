import sys

import cv2
import numpy as np

root_dir = '/Users/zhouzhenliang/Desktop/zlz/'

hist_channels = [0, 1, 2]
hist_size = [8, 8, 8]
hist_ranges = [0, 256, 0, 256, 0, 256]

x_count = 10
y_count = 14


def calc_hist(img):
    """
    直方图
    :param img:
    :return:
    """
    return cv2.calcHist([img], hist_channels, None, hist_size, hist_ranges)


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


def print_blocks():
    print()
    print('      0     1     2     3     4     5     6     7     8     9   ')
    print('   -------------------------------------------------------------')
    for y in range(0, y_count):
        print(f'{"%02d" % y} |', end='')
        for x in range(0, x_count):
            name = blocks[x][y]
            print(f' {"%3s" % name} |', end='')
        print()
        print('   -------------------------------------------------------------')


def detect_fill_blocks():
    """
    利用图像识别技术输出命名方块矩阵
    :return:
    """
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
            # path = f'{root_dir}/z_out_{i}-{j}.jpg'
            # cv2.imwrite(path, img)
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

    # 填充blocks
    print()
    group_index = 0
    for groups in groups_list:
        group_index += 1
        name = f'B{"%02d" % group_index}'
        print(f'{name}: ', end='')
        for result in groups:
            print(f'({result[0]},{result[1]}) ', end='')
            blocks[result[0]][result[1]] = name


def check_game_success():
    for x, y in travel_game_blocks():
        if len(blocks[x][y]) > 0:
            return False
    return True


def travel_game_blocks():
    for x in range(0, x_count):
        for y in range(0, y_count):
            yield x, y


def travel_game_active_blocks():
    for x, y in travel_game_blocks():
        if len(blocks[x][y]) > 0:
            yield x, y


def travel_game_same_blocks(x, y):
    for i, j in travel_game_active_blocks():
        if x == i and y == j:
            continue
        if blocks[x][y] == blocks[i][j]:
            yield i, j


# 水平方向是否挨着
def is_block2block_x_near(x1, y1, x2, y2):
    if y1 != y2:
        return False
    for x in range(min(x1, x2) + 1, max(x1, x2)):
        if len(blocks[x][y1]) > 0:
            return False
    return True


# 竖直方向是否挨着
def is_block2block_y_near(x1, y1, x2, y2):
    if x1 != x2:
        return False
    for y in range(min(y1, y2) + 1, max(y1, y2)):
        if len(blocks[x1][y]):
            return False
    return True


# 竖直移动，水平消除
def resolve_block2block_move_y(x1, y1, x2, y2):
    # 水平直接想通
    if is_block2block_x_near(x1, y1, x2, y2):
        return True, 0, []

    # 竖直方向移动也不可能水平相同
    if not is_block2block_x_near(x1, y2, x2, y2):
        return False, 0, None

    dy = y2 - y1
    if dy < 0:
        # 上移
        m_positions = [(x1, y1)]
        for y in range(y1 - 1, -1, -1):
            if len(blocks[x1][y]) == 0:
                break
            m_positions.append((x1, y))
        block = m_positions[-1]
        if block[1] + dy >= 0:
            return True, dy, m_positions[1:]
        else:
            return False, 0, None

    # 下移
    m_positions = [(x1, y1)]
    for y in range(y1 + 1, y_count):
        if len(blocks[x1][y]) == 0:
            break
        m_positions.append((x1, y))
    block = m_positions[-1]
    if block[1] + dy < y_count:
        return True, dy, m_positions[1:]
    else:
        return False, 0, None


# 水平移动，竖直消除
def resolve_block2block_move_x(x1, y1, x2, y2):
    # 竖直直接想通
    if is_block2block_y_near(x1, y1, x2, y2):
        return True, 0, []

    # 水平方向移动也不可能竖直相同
    if not is_block2block_y_near(x2, y1, x2, y2):
        return False, 0, None

    dx = x2 - x1
    if dx < 0:
        # 左移
        m_positions = [(x1, y1)]
        for x in range(x1 - 1, -1, -1):
            if len(blocks[x][y1]) == 0:
                break
            m_positions.append((x, y1))
        block = m_positions[-1]
        if block[0] + dx >= 0:
            return True, dx, m_positions[1:]
        else:
            return False, 0, None

    # 右移
    m_positions = [(x1, y1)]
    for x in range(x1 + 1, x_count):
        if len(blocks[x][y1]) == 0:
            break
        m_positions.append((x, y1))
    block = m_positions[-1]
    if block[0] + dx < x_count:
        return True, dx, m_positions[1:]
    else:
        return False, 0, None


actions = []


def show_blocks():
    print('--------------------------- show ---------------------------')
    print_blocks()
    w = 100
    h = 101
    img = np.empty((y_count * h, x_count * w, 3), np.uint8)
    for x, y in travel_game_active_blocks():
        path = f'{root_dir}/z_out_{x}-{y}.jpg'
        img[y * h:y * h + h, x * w:x * w + w] = cv2.imread(path)
    cv2.imshow('blocks', img)
    cv2.waitKey(0)


def start_game(deep):
    if check_game_success():
        print('--------------------------- game success ---------------------------')
        for action in actions:
            print(action)
        return True

    for x1, y1 in travel_game_active_blocks():
        for x2, y2 in travel_game_same_blocks(x1, y1):
            # 竖直移动
            result, dy, m_positions = resolve_block2block_move_y(x1, y1, x2, y2)
            if not result:
                continue
            action = f'({x1}, {y1}) - ({x2}, {y2})'
            print(action)
            actions.append(action)
            name = blocks[x1][y1]
            blocks[x1][y1] = ''
            blocks[x2][y2] = ''
            for mx, my in m_positions:
                blocks[mx][my + dy] = blocks[mx][my]
            show_blocks()
            if start_game(deep + 1):
                return True
            for mx, my in m_positions:
                blocks[mx][my] = blocks[mx][my + dy]
            blocks[x1][y1] = name
            blocks[x2][y2] = name
            actions.pop()
            show_blocks()

        for x2, y2 in travel_game_same_blocks(x1, y1):
            # 水平移动
            result, dx, m_positions = resolve_block2block_move_x(x1, y1, x2, y2)
            if not result:
                continue
            action = f'({x1}, {y1}) - ({x2}, {y2})'
            print(action)
            actions.append(action)
            name = blocks[x1][y1]
            blocks[x1][y1] = ''
            blocks[x2][y2] = ''
            for mx, my in m_positions:
                blocks[mx + dx][my] = blocks[mx][my]
            show_blocks()
            if start_game(deep + 1):
                return True
            for mx, my in m_positions:
                blocks[mx][my] = blocks[mx + dx][my]
            blocks[x1][y1] = name
            blocks[x2][y2] = name
            actions.pop()
            show_blocks()

    return False


if __name__ == '__main__':
    sys.setrecursionlimit(1000000)
    blocks = [['' for _ in range(0, y_count)] for _ in range(0, x_count)]
    detect_fill_blocks()
    print_blocks()
    start_game(0)
