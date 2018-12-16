import sys
import os
import time
import cPickle as pkl

# set your dataset location path here
ROOT_PATH = 'data'


def readPkl(filename):
    with open(filename, 'r') as f:
        data = pkl.load(f)
    return data


def writePkl(dictdata, filename):
    # {'errors':mean_all_errors}
    pkl.dump(dictdata, open(filename, 'wb'), -1)



def makedirsforfile(filename):
    try:
        os.makedirs(os.path.split(filename)[0])
    except:
        pass


def niceNumber(v, maxdigit=6):
    """Nicely format a number, with a maximum of 6 digits."""
    assert(maxdigit >= 0)

    if maxdigit == 0:
        return "%.0f" % v

    fmt = '%%.%df' % maxdigit
    s = fmt % v
    
    if len(s) > maxdigit:
        return s.rstrip("0").rstrip(".")
    elif len(s) == 0:
        return "0"
    else:
        return s

       

def checkToSkip(filename, overwrite):
    if os.path.exists(filename):
        print ("%s exists." % filename),
        if overwrite:
            print ("overwrite")
            return 0
        else:
            print ("skip")
            return 1
    return 0    
    
    
def printMessage(message_type, trace, message):
    print ('%s %s [%s] %s' % (time.strftime('%d/%m/%Y %H:%M:%S'), message_type, trace, message))

def printStatus(trace, message):
    printMessage('INFO', trace, message)

def printError(trace, message):
    printMessage('ERROR', trace, message)


class CmdOptions:
    def __init__(self):
        self.value = {}
        self.addOption("rootpath", ROOT_PATH)
        self.addOption("overwrite", 0)
        self.addOption("dryrun", 0)
        self.addOption("numjobs", 1)
        self.addOption("job", 1)

    def printHelp(self):
        print ("""
        --rootpath [default: %s]
        --numjobs [default: 1]
        --job [default: 1]
        --overwrite [default: %d]""" % (self.getString("rootpath"), self.getInt("overwrite")))
              
    def addOption(self, param, val):
        self.value[param] = val

    def getString(self, param):
        return self.value[param]

    def getInt(self, param):
        return int(self.getDouble(param))

    def getDouble(self, param):
        return float(self.getString(param))

    def getBool(self, param):
        return self.getInt(param) == 1

    def parseArgs(self, argv):
        i = 0
        while i < len(argv) -1:
            if argv[i].startswith("--"):
                if argv[i+1].startswith("--"):
                    i += 1
                    continue
                param = argv[i][2:]
                if param in self.value:
                    self.value[param] = argv[i+1]
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        okay = self.checkArgs()
        if not okay:
            self.printHelp()
        return okay


    def printArgs(self):
        for key in self.value.keys():
            print ("--%s %s" % (key, self.getString(key)))

    def checkArgs(self):
        paramsNeeded = [param for (param,value) in self.value.iteritems() if value is ""]

        if paramsNeeded:
            printError(self.__class__.__name__,"Need more arguments: %s" % " ".join(paramsNeeded))
            return False

        if self.getInt("numjobs") < self.getInt("job"):
            printError(self.__class__.__name__, "numjobs cannot be smaller than job")
            return False

        return True


def total_seconds(td):
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 1e6) / 1e6

if __name__ == "__main__":
    cmdOpts = CmdOptions()
    cmdOpts.printHelp()
    cmdOpts.parseArgs(sys.argv[1:])
    print niceNumber(1.0/3, 4)
    for i in range(0, 15):
        print niceNumber(8.17717824342e-10, i)
        
        
