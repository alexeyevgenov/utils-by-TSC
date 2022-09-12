import logging
# first of all below we have commands to set up logger - it logs the stuff to the console now, and
# also writes to the file 'debug.txt'. All the logged places can be found in the script with 'log. ... ' command
# it can be a convenient starting point for collecting all the needed logs and info - basically
# just similar commands will be needed once you know what type of info you want to collect

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s, %(message)s',
                    filename='debug.log'
                    )

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.getLogger('lib_name').setLevel(logging.WARNING)

log = logging.getLogger(__name__)


# using the same log in another class
class Example:
    def __init__(self):
        self.logger = logging.getLogger(__class__.__name__)
