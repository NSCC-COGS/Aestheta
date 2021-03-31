import os
import sys

from pathlib import Path

import toml

class Placemark:
    _placemarks = {}
    _aliases = {}
    
    @classmethod
    def get(self, term):
        return self._placemarks.get(term) or self._aliases.get(term)
    
    # Load Placemark config from a directory.
    @classmethod
    def load_config(self, base_path='Config/placemarks'):
        self._placemarks = {}
        self._aliases = {}
        
        # Note our current directory
        previous_dir = os.getcwd()
        
        # Change our directory to the base path
        os.chdir(base_path)

        # List the contents of the directory, and convert each entry to a
        # pathlib.Path() to make it easier to work with.
        #
        # Take only the paths that are files and end with '.toml'.
        paths = [
            path
            for path in [
                Path(entry)
                for entry in os.listdir()
            ]
            if path.is_file() and path.suffix == '.toml'
        ]
        
        placemarks = []
        for path in paths:
            try:
                # Try to load the placemark from the file and append it to
                # our list of placemarks.
                placemark = self.load_file(path)
                placemarks.append(placemark)
            except toml.decoder.TomlDecodeError:
                # If the toml library can't parse this file, it's probably not
                # valid. Print a warning and skip this file.
                print(
                    f'{repr(self)} ignored {path} because it is not a valid'
                    f' TOML file.',
                    file=sys.stderr
                )
            except ValueError as error:
                # This could mean that the placemark didn't have a valid lat and
                # lon. Print a warning and skip this file.
                print(
                    f'{repr(self)} ignored {path} because {error}',
                    file=sys.stderr
                )
                
        for placemark in placemarks:
            self._placemarks[placemark.name] = placemark
            for alias in placemark.aliases:
                self._aliases[alias] = placemark

        # Change our directory back
        os.chdir(previous_dir)
        
    @classmethod
    def load_file(self, path):
        path2 = Path(path)
        
        # In the case of a file named eiffel_tower.toml, eiffel_tower is the
        # stem, and .toml is the suffix. Use the stem as the 'official' name of
        # this placemark.
        name = path2.stem
        with open(str(path2)) as f:
            return self.load(f, name)
        
    @classmethod
    def load(self, f, name):
        # Use the toml library to read the file into config. A TOML file
        # with this text:
        #
        # description = 'NSCC Centre of Geographic Sciences'
        # lat = 44.88514830522117
        # lon = -65.1684601455184
        #
        # Gets turned into a Python dict (key/value store) that looks like this:
        #
        # {
        #     'description': 'NSCC Centre of Geographic Sciences',
        #     'lat': 44.88514830522117,
        #     'lon': -65.1684601455184
        # }
        #
        # And we assign that to config.
        config = toml.load(f)
        
        # Make a new placemark with the config we got from the file.
        #
        # **config breaks config out into keyword arguments. Using the sample
        # config above, we'd get:
        #
        # placemark = self(
        #     name=name,
        #     description='NSCC Centre of Geographic Sciences',
        #     lat=44.88514830522117,
        #     lon=-65.1684601455184
        # )
        #
        # self refers to this class, which is Placemark. Placemark(name=name,
        # etc.) creates a new Placemark. The arguments to this function go to
        # the __init__ function, which prepares a placemark, which is initially
        # blank, for use.
        # 
        # If __init__ is successful, we get a new Placemark, ready to use, and
        # can assign that to our placemark variable.
        placemark = self(name=name, **config)
        
        return placemark
        
    # When we create a new Placemark object, Python calls its __init__ method
    # (function) to initialize it.
    #
    # self is this object. Allows the object to refer to itself. Python will
    # give self as the first argument to any function (method) you define in a
    # class.
    def __init__(self, name='', description='',
            lat=None, lon=None, default_zoom=None, aliases = []):
        self.name = name
        self.description = description
        
        # Make sure we have both latitude and longitude!
        if isinstance(lat, float) and isinstance(lon, float):
            self.xy = (lon, lat)
        else:
            # Raise a ValueError exception if we don't.
            raise ValueError(
                f'{repr(self)} expects floating point values'
                f' for both lat and lon, but got {repr(lat)} and {repr(lon)}.'
            )
        
        self.default_zoom = default_zoom
        self.aliases = aliases
        
    # This overrides what we see when we inspect a Placemark in an interactive
    # Python session. For example, when we do this:
    #
    # >>> placemark = Placemark(name='eiffel_tower', description, etc.)
    # >>> placemark
    #
    # Python will use the repr() function, which calls __repr__(), to determine
    # what to display. For example,
    #
    # <Placemark 'eiffel_tower' (2.2945, 48.8584)>
    def __repr__(self):
        class_name = type(self).__name__
        return f'<{class_name} {repr(self.name)} {repr(self.xy)}>'