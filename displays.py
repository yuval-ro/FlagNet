import numpy as np
import torch

COLOR_ALIAS = [ 'red', 'r', 'red Bold', 'rB',
                'green', 'g', 'green Bold', 'gB',
                'yellow', 'y', 'yellow Bold', 'yB',
                'blue', 'b', 'blue Bold', 'bB']
STR_FORMAT = '{0:<{1}}'
WIDTH = 20

class Style:
	END =       '\033[0m'
	DEFAULT =	'\033[0m'
	BOLD = 		'\033[1m'
	BLACK = 	'\033[30m'
	RED =		'\033[31m'
	GREEN =		'\033[32m'
	YELLOW =	'\033[33m'
	BLUE =		'\033[34m'
	MAGENTA =	'\033[35m'
	CYAN =		'\033[36m'
	WHITE =		'\033[37m'

def color(clr: str = None) -> str:
	if clr == None:
		return Style.END
	if clr not in COLOR_ALIAS:
		raise SystemExit(f'color() got an unfamillier color code \'{clr}\'')
	if clr in ['red', 'r', 'rB']:
		return Style.RED if color != 'rB' else Style.BOLD + Style.RED
	if clr in ['green', 'g', 'gB']:
		return Style.GREEN if color != 'gB' else Style.BOLD + Style.GREEN
	if clr in ['yellow', 'y', 'yB']:
		return Style.YELLOW if color != 'yB' else Style.BOLD + Style.YELLOW
	if clr in ['blue', 'b', 'bB']:
		return Style.BLUE if color != 'bB' else Style.BOLD + Style.BLUE
	pass

def print_line(inputs: list,
    width: int = WIDTH) -> None:
    VALID_TYPES = [str, float, int]
    # correct = lambda x: FLOAT_FORMAT if isinstance(x, float) else STR_FORMAT
    correct = lambda x: STR_FORMAT
    for idx, item in enumerate(inputs):
        if any(isinstance(item, t) for t in VALID_TYPES):
            print(
                Style.DEFAULT +
                correct(item).format(item, width),
                end=''
                )
        elif isinstance(item, tuple) and len(item) == 2:
            s, clr = item
            print(
                color(clr) +
                correct(s).format(s, width),
                end=''
                )
        else:
            raise SystemExit(f'unknown argument passed to print_line(): \'{item}\'')
        if idx == (len(inputs) - 1):
            print(Style.END)

def print_header(inputs: list[str],
    width: int = WIDTH) -> None:
    for idx, s in enumerate(inputs):
        print(Style.BOLD + STR_FORMAT.format(s, width), end='')
        if idx == (len(inputs) - 1):
            print(Style.END)

def chunkifier(items: list,
    size: int) -> list:
  for i in range(0, len(items), size):
    yield items[i : i+size]

def print_matrix(items: list[tuple],
    rows: int = None,
    cols: int = None,
    vector: bool = False) -> None:
    flag = False
    if rows == None and cols == None:
        if vector == True:
            rows = len(items)
            cols = 1
        else:
            cols = len(items)
            rows = 1

    # rows supplied, cols not:
    elif rows != None and cols == None:
        if rows > len(items) or rows <= 0:
            raise SystemExit('bad dim')
        cols = int(np.ceil(len(items) / rows))
    # cols supplied, rows not:
    elif rows == None and cols != None:
        if cols > len(items) or cols <= 0:
            raise SystemExit('bad dim')
        rows = int(np.ceil(len(items) / cols))
        flag = True
    # both supplied:
    if any([rows * cols < len(items),
            cols >= len(items) and rows > 1,
            rows >= len(items) and cols > 1]):
            raise SystemExit('bad dim')
            
    if flag: # split to n-sized chunks
        x = chunkifier(items, cols)
    else: # split to n chunks
        x = np.array_split(items, rows)
    for row in x:
        headers = []
        data = []
        for tup in row:
            headers.append(tup[0])
            data.append(tup[1])
        print_header(headers)
        print_line(data)
        print()

def print_msg(msg: str,
    clr: str = 'y') -> None:
	print(color(clr) + msg + color())

def seconds_to_time(seconds: float,
    hrs: bool = False):
    if hrs:
        return '%02d:%02d:%02d'%((seconds // 3600), (seconds // 60), (round(seconds % 60)))
    else:
        return '%02d:%02d'%((seconds // 60), (round(seconds % 60)))

# Defining a routine for displaying a single epoch's metadata:
def print_epoch(data: list[tuple],
    i: int) -> None:
    print_line([
        data[i][0],
        seconds_to_time(data[i][1]),
        data[i][2],
        data[i][3],
        data[i][4]
    ])

# Defining a routine for displaying a summary of a training session's collected metadata:
def print_train_summary(data: list[tuple]) -> None:
    print_matrix([
        ('epochs',          len(data)),
        ('total time',      seconds_to_time(np.sum([item[1] for item in data])),),
        ('mean train loss', np.mean([item[2] for item in data])),
        ('mean valid loss', np.mean([item[2] for item in data])),
        ('mean accuracy',   np.mean([item[4] for item in data])),
    ])

# Defining a routine for displaying a single testing pass metadata,):
def print_test_summary(loss: float,
    acc: float,
    loader: torch.utils.data.DataLoader) -> None:
    print_matrix([
        ('',        ''),
        ('',        ''),
        ('loss',    loss / len(loader)),
        ('',        ''),
        ('accuracy', acc / len(loader))
    ])