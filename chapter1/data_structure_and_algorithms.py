# 命名切片
def slice_test():
    #         00000000001111111111222222222233333333334444444444
    #         01234567890123456789012345678901234567890123456789
    record = '..............100........513.25.............'
    print(record[14:17])
    print(record[25:31])


if __name__ == '__main__':
    slice_test()
