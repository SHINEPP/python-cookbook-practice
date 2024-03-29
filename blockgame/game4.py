import cv2
import numpy as np

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


def detect_fill_blocks():
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
            blocks[result[0]][result[1]][0] = name
        print()

    print()
    print('      0     1     2     3     4     5     6     7     8     9   ')
    print('   -------------------------------------------------------------')
    for y in range(0, y_count):
        print(f'{"%02d" % y} |', end='')
        for x in range(0, x_count):
            name = blocks[x][y][0]
            print(f' {name} |', end='')
        print()
        print('   -------------------------------------------------------------')


def game_check_success():
    for x_blocks in blocks:
        for block in x_blocks:
            if block[1]:
                return False
    return True


def game_travel_active_blocks():
    for y in range(0, y_count):
        for x in range(0, x_count):
            if blocks[x][y][1]:
                yield x, y


def game_can_move_block(x, y, dx, dy):
    if dx == 0 and dy == 0:
        return True
    x1 = x + dx
    y1 = y + dy
    if x1 < 0 or x1 >= x_count or y1 < 0 or y1:
        return False
    if not blocks[x1][y][1]:
        return True
    return game_can_move_block(x1, y1, dx, dy)


def game_move_block(x, y, dx, dy):
    if dx == 0 and dy == 0:
        return
    x1 = x + dx
    y1 = y + dy
    game_move_block(x)


def game_find_same_block(x, y):
    position = []
    name = blocks[x][y][0]
    for i in range(x + 1, x_count):
        if not blocks[i][y][1]:
            continue
        if name == blocks[i][y][0]:
            position.append((i, y))
        break
    for j in range(y + 1, y_count):
        if not blocks[x][j][1]:
            continue
        if name == blocks[x][j][0]:
            position.append((x, j))
        break
    return position


g_steps = []


def game_start():
    if game_check_success():
        print('------------- success ---------------')
        for p in g_steps:
            print(p)
        return True

    can_forward = False
    for x, y in game_travel_active_blocks():
        positions = game_find_same_block(x, y)
        if len(positions) > 0:
            can_forward = True
            break

    if not can_forward:
        print('------------- can\'t forward ---------------')
        for p in g_steps:
            print(p)
        return False

    for x, y in game_travel_active_blocks():
        for dx in range(0, x_count - x):
            for dy in range(0, y_count - y):
                can_move = game_can_move_block(x, y, dx, dy)

                positions = game_find_same_block(x, y)
                for x1, y1 in positions:
                    step = f'({x},{y}) -> ({x1},{y1})'
                    g_steps.append(step)
                    blocks[x][y][1] = False
                    blocks[x1][y1][1] = False
                    if game_start():
                        return True
                    blocks[x][y][1] = True
                    blocks[x1][y1][1] = True
                    g_steps.pop()


    return False


if __name__ == '__main__':
    # block: [x][y][0:name, 1:active]
    blocks = []
    for _ in range(0, x_count):
        blocks.append([['', True] for j in range(0, y_count)])

    detect_fill_blocks()

    game_start()
