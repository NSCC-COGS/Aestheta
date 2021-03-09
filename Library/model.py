# Standard library imports
import os
import pathlib
import pickle

from datetime import datetime

# Third-party imports

# Local imports
from . import environment

class InvalidModelNameError(Exception):
    pass

class InvalidModelError(Exception):
    pass
    
class UnsupportedModelArchitectureError(Exception):
    pass

class ModelName:
    @classmethod
    def parse(self, path):
        path = pathlib.Path(path)
        
        try:
            kind, timestamp_str, bitness_str, label = path.stem.split('_', 3)
        except ValueError:
            raise InvalidModelNameError
        
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
        bitness = int(bitness_str)
        suffix = path.suffix
        
        return self(kind=kind, timestamp=timestamp, bitness=bitness, label=label, suffix=suffix)
        
    @classmethod
    def latest_in_path(self, path):
        base_path = pathlib.Path(path)
        bitness = environment.bitness()

        names = []
        for p in os.listdir(path):
            try:
                name = self.parse(p)
                names.append(name)
            except InvalidModelNameError:
                pass
                
        names_with_bitness = [n for n in names if n.bitness == bitness]
        
        if len(names_with_bitness) > 0:
            name_of_latest = max(names_with_bitness, key=lambda n: n.timestamp)
            return name_of_latest
        else:
            return None
        
    def __init__(self, kind='simpleClassifier', timestamp=None, bitness=None, label='', suffix='.aist'):
        self.kind = kind
        
        self.timestamp = timestamp if timestamp else datetime.now()
        self.bitness = bitness if bitness else environment.bitness()
            
        self.label = label
        self.suffix = suffix
        
        name_components = [
            self.kind,
            self.timestamp.strftime('%Y%m%d%H%M%S'),
            str(self.bitness),
        ]
        
        if self.label:
            name_components.append(label)
        
        self.filename = '_'.join(name_components) + suffix

    def __str__(self):
        return self.filename
        
    def __repr__(self):
        return str(self)

class Model:
    @classmethod
    def load_named_or_latest(self, filename=None, dir_path='Models'):
        filename = filename if filename else ModelName.latest_in_path(dir_path).filename
        path = pathlib.Path(dir_path) / filename
        return self.load_from_path(path)

    @classmethod
    def load_from_path(self, path):
        # Raises FileNotFoundError if file path doesn't exist.
        with open(str(path), 'rb') as f:
            return self.load(f)
                
    @classmethod
    def load(self, f):
        try:
            model, classes = pickle.load(f)
            return self(model=model, classes=classes)
        except EOFError as error:
            raise InvalidModelError(f'Stream does not contain a valid model.') from error
        except pickle.UnpicklingError as error:
            raise InvalidModelError(f'Stream does not contain a valid model.') from error
        except ValueError as error:
            if 'Buffer dtype mismatch' in str(error):
                bitness = environment.bitness()
                raise UnsupportedModelArchitectureError(f'Model was not built for a {bitness}-bit Python environment.') from error
            else:
                raise error
                
    def __init__(self, model=None, classes=[], name=None):
        self.model = model
        self.classes = classes
        self.name = name
        
    def __str__(self):
        return f'{self.model}\n{self.classes}'
        
    def __repr__(self):
        return str(self)
                
    def dump(self, f):
        pickle.dump([self.model, self.classes], f)
        
    def save(self, label=None, dir_path='Models'):
        base_path = pathlib.Path(dir_path)
        name = ModelName(label=label)
        
        path = base_path / name.path
        with open(str(path), 'wb') as f:
            self.dump(f)
            
if __name__ == '__main__':
    print(Model.load_named_or_latest(filename='simpleClassifier_20210302180953_64_HelloEarth100.aist'))
    print(ModelName.latest_in_path('/'))