"""
utils
Usage:
    utils -h | --help

Options:
    -o FILE --output=FILE       Path to save results.
    -h --help                   Show this screen.
"""

from docopt import docopt


def main():
    """Main utility function."""
    options = docopt(__doc__)
