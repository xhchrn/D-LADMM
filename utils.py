import logging

def setup_logger(log_file):
    if log_file is not None:
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
        lgr = logging.getLogger()
        lgr.addHandler(logging.StreamHandler())
        lgr = lgr.info
    else:
        lgr = print

    return lgr

def my_str(obj):
    return '{:f}'.format(obj)

