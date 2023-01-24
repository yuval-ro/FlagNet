# Display functions definitions

import numpy as np

class Style:
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

def color(color, bold=False):
	if color in ['d', 'default']:
		return (Style.DEFAULT if not bold else Style.BOLD+Style.DEFAULT)
	if color in ['r', 'red']:
		return (Style.RED if not bold else Style.BOLD+Style.RED)
	if color in ['g', 'green']:
		return (Style.GREEN if not bold else Style.BOLD+Style.GREEN)
	if color in ['y', 'yellow']:
		return (Style.YELLOW if not bold else Style.BOLD+Style.YELLOW)
	else:
		raise SystemExit(f'passed unfamilier parameters passed to color()')

def println(strings, width=20, header=False):
	if header: # header lines
		for idx, s in enumerate(strings):
			if isinstance(s, tuple):
				SystemExit(f'passed a tuple while header=True')
			else:
				print(Style.BOLD+('{0:<{1}}').format(str(s), width), end='')
			if idx == (len(strings) - 1):
				print(color('d'))
	else:
		for idx, s in enumerate(strings):
			if isinstance(s, tuple): # color or bold applied
				print((color(s[1])+('{0:<{1}.3f}').format((s[0]), width) if isinstance(s[0], float) else color(s[1])+('{0:<{1}}').format(str(s[0]), width)), end='')

			else: # no color or bold needed
				print(color('d')+(('{0:<{1}.3f}').format(s, width) if isinstance(s, float) else color('d')+('{0:<{1}}').format(str(s), width)), end='')

			if idx == (len(strings) - 1):
				print(color('d'))

def seconds_to_time(seconds, hrs=False):
    if hrs:
        return '%02d:%02d:%02d'%((seconds // 3600), (seconds // 60), (round(seconds % 60)))
    else:
        return '%02d:%02d'%((seconds // 60), (round(seconds % 60)))

# Defining a routine for displaying a single epoch's metadata,
#  or an entire epoch list's metadata (depends if param 'idx' was supplied):
def displayTrain(list, idx=None):
    if idx is not None:
        println([list[idx][0],
                seconds_to_time(list[idx][1]),
                list[idx][2],
                list[idx][3],
                list[idx][4]])
    else:
        println(['epochs',
                'total time',
                'mean train loss',
                'mean valid loss',
                'mean accuracy'], header=True)
        println([len(list),
                seconds_to_time(
                np.sum([item[1] for item in list])
                ),
                np.mean([item[2] for item in list]),
                np.mean([item[3] for item in list]),
                np.mean([item[4] for item in list]),
            ])

# Defining a routine for displaying a single testing pass metadata,):
def displayTest(loss, acc, loader):
    println(['', '', 'loss', '', 'accuracy'], header=True)
    println(['', '', (loss / len(loader)), '', (acc / len(loader))])