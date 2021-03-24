"""
utils
Usage:
    utils explode <video> [-n NUM]
    utils implode <folder> [--fps VAL]
    utils -h | --help

Options:
    --fps VAL                   Number of frame per second [default: 25].
    -n NUM --num=NUM            Number of frames.
    -o FILE --output=FILE       Path to save results.
    -h --help                   Show this screen.
"""
import os
from glob import glob

import cv2
from docopt import docopt
from tqdm import tqdm


def main():
    """Main utility function."""
    options = parse_arguments(docopt(__doc__))

    if options['explode']:
        command_explode(options)
    elif options['implode']:
        command_implode(options)


def command_explode(options):
    # Open capture.
    cap = cv2.VideoCapture(options['<video>'])

    # Define destination folder name.
    video_name = os.path.splitext(os.path.basename(options['<video>']))[0]
    dst_folder = os.path.join('data', f'{video_name}_exploded')

    # Create/empty destination folder.
    os.makedirs(dst_folder, exist_ok=True)
    for file in os.listdir(dst_folder):
        os.remove(os.path.join(dst_folder, file))

    # Set the limit to frames to read.
    limit = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if options['--num']:
        limit = min(limit, options['--num'])

    # Write frames to images.
    for i in tqdm(range(limit)):
        _, frame = cap.read()
        frame_name = os.path.join(dst_folder, f'{str(i).zfill(6)}.jpg')
        cv2.imwrite(frame_name, frame)

    # Release capture.
    cap.release()


def command_implode(options):
    frames = glob(os.path.join(options['<folder>'], '*.jpg'))
    frames = sorted(frames)

    # Return if no frame is available.
    if len(frames) == 0:
        return

    # Get frame size from a sample image.
    sample = cv2.imread(frames[0])
    frame_size = sample.shape[1::-1]

    # Define destination file name.
    folder_name = os.path.basename(options['<folder>'])
    dst_file = os.path.join('data', f"{folder_name.split('_')[0]}_imploded.avi")

    # Open capture.
    out = cv2.VideoWriter(dst_file, cv2.VideoWriter_fourcc(*'DIVX'), options['--fps'], frame_size)

    # Write frames to video.
    progress_bar = tqdm(frames)
    for frame in progress_bar:
        img = cv2.imread(frame)
        out.write(img)

    # Release capture.
    out.release()


def parse_arguments(options):
    if options['--fps']:
        options['--fps'] = float(options['--fps'])

    if options['--num']:
        options['--num'] = int(options['--num'])

    if options['<folder>']:
        if options['<folder>'].endswith('/'):
            options['<folder>'] = options['<folder>'][:-1]

        if not os.path.exists(options['<folder>']):
            raise FileNotFoundError(f"{options['<folder>']} was not found.")

        if not os.path.isdir(options['<folder>']):
            raise NotADirectoryError(f"{options['<folder>']} is not a directory.")

    if options['<video>']:
        if not os.path.exists(options['<video>']):
            raise FileNotFoundError(f"{options['<video>']} was not found.")

        if os.path.isdir(options['<video>']):
            raise IsADirectoryError(f"{options['<video>']} is a directory.")

    return options
