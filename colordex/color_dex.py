import getopt
import sys


def percent_dex(percent):
    value = percent / 100.0 * 255
    return '%d = 0x%02X' % (percent, round(value))


if __name__ == '__main__':
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, '-d:')
    for opt_name, opt_value in opts:
        if opt_name == '-d':
            print(percent_dex(int(opt_value)))
