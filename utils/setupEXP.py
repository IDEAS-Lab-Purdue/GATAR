import tensorboard
# import summaryWriter
from torch.utils.tensorboard import SummaryWriter
import logging

def start(path):

    # init logger and tensorboard
    

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(path + '/log.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # init tensorboard
    tb = SummaryWriter(path + '/tensorboard')

    return logger, tb
