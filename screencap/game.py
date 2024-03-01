import os
from datetime import datetime

if __name__ == '__main__':
    now = datetime.now()
    ftime = now.strftime('%Y%m%d%H%M%S-%f')
    phone_dir = '/sdcard/adb_screencap_temp'
    local_dir = '/Users/zhouzhenliang/temp'
    os.system(f'adb shell mkdir {phone_dir}')
    os.system(f'adb shell screencap -p {phone_dir}/{ftime}.png')
    os.system(f'adb pull {phone_dir}/{ftime}.png {local_dir}')
    os.system(f'adb shell rm -f {phone_dir}/{ftime}.png')
