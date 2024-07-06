import os
from .color import color
import ping3
from datetime import datetime


def find_color(color_):
    if color_ == "red":
        return color.RED
    elif color_ == "yellow":
        return color.YELLOW
    elif color_ == "green":
        return color.GREEN
    elif color_ == "cyan":
        return color.CYAN
    elif color_ == "purple":
        return color.PURPLE
    elif color_ == "blue":
        return color.BLUE
    else:
        return ""


def cur_time_str():
    cur_time = datetime.now()
    cur_time_string = "[" + cur_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + "] "
    return cur_time_string


def print_log(line, color_="green", show_time=True):
    if type(line) == str:
        color_str = find_color(color_)
        if show_time:
            print(color.PURPLE + cur_time_str(), end=color.END)
        else:
            line = "           " + line
        print(color_str + line + color.END)
    else:
        print(line)


def int_to_ubyte(num):
    return num.to_bytes(1, "big", signed=False)


def int_to_Nubyte(num, N):
    return num.to_bytes(N, "big", signed=False)


def choose_file_in_folder_by_order(folder, file_order):
    dir_name = folder
    # Get list of all files in a given directory sorted by name
    list_of_files = sorted(filter(lambda x: os.path.isfile(os.path.join(dir_name, x)),
                                  os.listdir(dir_name)))
    return list_of_files[file_order]


def ping_host(host, count=10):
    ping_result = [ping3.ping(host) for _ in range(count)]
    ping_result = [result for result in ping_result if
                   result is not None]  # Loại bỏ các kết quả None (không thành công)

    if ping_result:
        avg_latency = sum(ping_result) / len(ping_result)
        min_latency = min(ping_result)
        max_latency = max(ping_result)
        packet_loss = (1 - len(ping_result) / count) * 100
    else:
        avg_latency = None
        min_latency = None
        max_latency = None
        packet_loss = 100

    return {
        'host': host,
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'packet_loss': packet_loss
    }
