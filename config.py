import configparser


class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))

        # Parameter
        self.fdim = conf.getint("Data_Setting", "fdim")
        self.k = conf.getint("Model_Setup", "k")
        self.radius = conf.getint("Model_Setup", "radius")
        self.seed = conf.getint("Model_Setup", "seed")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.epochs = conf.getint("Model_Setup", "epochs")
        self.no_cuda = conf.getboolean("Model_Setup", "no_cuda")
        self.no_seed = conf.getboolean("Model_Setup", "no_seed")


class Config_HBC(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))

        # Parameter
        self.fdim = conf.getint("Data_Setting", "fdim")
        self.k = conf.getint("HBC_Model_Setup", "k")
        self.radius = conf.getint("HBC_Model_Setup", "radius")
        self.seed = conf.getint("HBC_Model_Setup", "seed")
        self.lr = conf.getfloat("HBC_Model_Setup", "lr")
        self.weight_decay = conf.getfloat("HBC_Model_Setup", "weight_decay")
        self.epochs = conf.getint("HBC_Model_Setup", "epochs")
        self.no_cuda = conf.getboolean("HBC_Model_Setup", "no_cuda")
        self.no_seed = conf.getboolean("HBC_Model_Setup", "no_seed")


class Config_MBA(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))

        # Parameter
        self.fdim = conf.getint("Data_Setting", "fdim")
        self.k = conf.getint("MBA_Model_Setup", "k")
        self.radius = conf.getint("MBA_Model_Setup", "radius")
        self.seed = conf.getint("MBA_Model_Setup", "seed")
        self.lr = conf.getfloat("MBA_Model_Setup", "lr")
        self.weight_decay = conf.getfloat("MBA_Model_Setup", "weight_decay")
        self.epochs = conf.getint("MBA_Model_Setup", "epochs")
        self.no_cuda = conf.getboolean("MBA_Model_Setup", "no_cuda")
        self.no_seed = conf.getboolean("MBA_Model_Setup", "no_seed")