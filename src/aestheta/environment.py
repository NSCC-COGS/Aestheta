import struct
#import toml

def bitness():
    return struct.calcsize("P") * 8
    
#def load_config():
#    with open('Config/config.toml') as f:
#        return toml.load(f)