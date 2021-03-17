import functools
import math
import os
import string
import sys
import urllib.parse

from pathlib import Path

import imageio
import numpy as np
import requests
import toml

from placemark import Placemark

class TileSource:
    # The HTTP User-Agent header identifies the web browser, which can allow the
    # server to dish out content that's appropriate for that browser, or log
    # what kind of devices people are using to browse.
    #
    # Here, for development purposes, we're pretending to be a regular web
    # browser.
    HTTP_REQUEST_HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4)'
            ' AppleWebKit/537.36 (KHTML, like Gecko)'
            ' Chrome/83.0.4103.97 Safari/537.36'
        )
    }
        
    sources = {}
    aliases = {}
    
    @classmethod
    def get(self, term):
        return self.sources.get(term)
        
    @classmethod
    def get_alias(self, term):
        return self.aliases.get(term)
        
    # Build a set of TileSources from a directory of config files.
    #
    # This is a class method. Instead of applying to TileSource objects, it
    # applies to the TileSource class. We can call it like this:
    # TileSource.load_config(base_path=<whatever>).
    @classmethod
    def load_config(self, base_path='Config/tile_sources'):
        # Clear our current source and alias dicts.
        self.sources = {}
        self.aliases = {}
        
        # Note our current directory
        previous_dir = os.getcwd()
        
        # Change our directory to the base path
        os.chdir(base_path)

        # List the contents of the directory, and convert each entry to a
        # pathlib.Path() to make it easier to work with.
        #
        # Take only the paths that are files and end with '.toml'.
        #
        # This is a list comprehension, which is a bit like an SQL query. If
        # this were an SQL query, it would look like:
        #
        # SELECT
        #     path
        # FROM
        #     (
        #       SELECT
        #           Path(entry) AS path
        #       FROM
        #           os_listdir()
        #     )
        # WHERE
        #     is_file(path) and suffix(path) = '.toml'
        paths = [
            path
            for path in [
                Path(entry)
                for entry in os.listdir()
            ]
            if path.is_file() and path.suffix == '.toml'
        ]
        
        sources = []
        for path in paths:
            try:
                source = self.load_file(path)
                sources.append(source)
            except toml.decoder.TomlDecodeError:
                print(
                    f'{repr(self)} ignored {path} because it is not a valid'
                    f' TOML file.',
                    file=sys.stderr
                )
                
        for source in sources:
            self.sources[source.name] = source
            
            # Aliases let us refer to tile sources and layers by their
            # original names.
            for alias, attrs in source.aliases.items():
                self.aliases[alias] = (source, attrs.get('layer'))
                
        # Change our directory back
        os.chdir(previous_dir)
        
    @classmethod
    def load_file(self, path):
        path2 = Path(path)
        name = path2.stem
        with open(str(path2)) as f:
            return self.load(f, name)
        
    @classmethod
    def load(self, f, name):
        config = toml.load(f)
        source = self(name=name, **config)
        return source
            
    def __init__(self, name='', description='', url_template='',
            access_token='', layer_key={}, aliases={}):
        self.name = name
        self.description = description
        self.url_template = url_template
        self.access_token = access_token
        self.layer_key = layer_key
        self.aliases = aliases
        
        if not url_template:
            raise ValueError(
                f'{repr(self)} needs a URL template, but'
                f' {repr(url_template)} is given.'
            )
        
        # Use urllib.parse.urlparse to break the URL into its constituent parts.
        url_parse_result = urllib.parse.urlparse(url_template)
        
        # In https://google.ca, 'https' is the scheme. Warn if the URL template
        # specifies 'http', which isn't secure.
        if url_parse_result.scheme != 'https':
            print(
                f'Caution! The URL for {repr(self)} uses'
                f' {url_parse_result.scheme}, but should use https.',
                file=sys.stderr
            )
        
        # Use string.Formatter() to identify the variables we need to substitute
        # in the URL template, e.g., {x}, {y}, {layer}.
        formatter = string.Formatter()
        url_template_keys = [
            match[1] for match in formatter.parse(url_template)
        ]
        
        # Does the template specify an {access_token}? What about {layer}?
        self.uses_access_token = 'access_token' in url_template_keys
        self.uses_layers = 'layer' in url_template_keys
        
        # If the URL template needs an access token but no access token was
        # given (or it's empty), this isn't going to work. Raise an exception.
        if self.uses_access_token:
            if not access_token:
                raise ValueError(
                    f'URL template for {repr(self)} expects an access token,'
                    f' but none is given.'
                )
                
            # Partially apply the format function, with just the access token.
            # We'll supply the rest of the details later.
            self.url_partial = functools.partial(
                url_template.format, access_token=access_token
            )
        else:
            # Partially apply the format function, with no details. We'll supply
            # the details later.
            self.url_partial = functools.partial(url_template.format)
            
        # If the URL template needs a layer but no layers were given, or the
        # layer key is empty, this isn't going to work. Raise an exception.
        if self.uses_layers and not layer_key:
            raise ValueError(
                f'URL template for {repr(self)} expects layers, but no layer'
                f' key is given.'
            )
            
    # The following two functions are adapted from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Lon..2Flat._to_tile_numbers_2        
    
    # Get a tile (x, y) index from a lat, long and zoom level.
    def xy_from_lon_lat(self, lon_lat=(0.0, 0.0), zoom=0):
        lon, lat = lon_lat
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
        
    # Get a (lon, lat) coordinate pair from a tile x, y and zoom level.
    def lon_lat_from_xy(self, xy=(0, 0), zoom=0):
        x, y = xy
        n = 2.0 ** zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        return (lon_deg, lat_deg)
        
    def location_zoom_from_xyz(self, xyz=[0, 0, 0]):
        location, zoom = (0, 0), 0
        try:
            # Assume xyz has three elements: x (or lon), y (or lat), and zoom.
            x, y, zoom = xyz
            location = (x, y)
        except ValueError:
            try:
                # Assume xyz has two elements: location, e.g. 'eiffel', and
                # zoom.
                location, zoom = xyz
            except ValueError:
                # Assume xyz is one element: location.
                location = xyz
                zoom = None
                
        return location, zoom
        
    def xy_zoom_from_location(self, location=(0, 0), zoom=None):
        if isinstance(location, str):
            try:
                placemark = Placemark.get(location)
                x, y = placemark.xy
                zoom2 = zoom if zoom != None else placemark.default_zoom
            except AttributeError as error:
                raise ValueError(
                    f'Could not find a placemark for {repr(location)}'
                ) from error
        elif isinstance(location, Placemark):
            x, y = location.xy
            zoom2 = zoom if zoom != None else location.default_zoom
        else:
            x, y = location
            zoom2 = zoom
            
        zoom3 = zoom2 if zoom2 != None else 15
        
        if isinstance(x, float) and isinstance(y, float):
            x, y = self.xy_from_lon_lat((x, y), zoom3)
                
        return (x, y), zoom3
        
    def xy_zoom_from_xyz(self, xyz=[0, 0, 0]):
        location, zoom = self.location_zoom_from_xyz(xyz)
        return self.xy_zoom_from_location(location, zoom)
            
    # Get a single tile at a given x/y (can be tile coords or lon, lat), at
    # a given zoom level (0-15), and a given layer (e.g. 'street' or
    # 'satellite'), required if the tile source uses layers.
    def tile(self, location=(0, 0), zoom=None, layer=None):
        if isinstance(location, list):
            xy, zoom2 = self.xy_zoom_from_xyz(location)
        else:
            xy, zoom2 = self.xy_zoom_from_location(location, zoom)
            
        if __debug__:
            print(
                f'Tile requested for location {repr(location)} zoom {zoom}'
                f' from {repr(self)} layer {repr(layer)}.'
            )
            
        # Get the URL for this tile.
        url = self.url(xy, zoom2, layer)
        if __debug__:
            print(
                f'Getting tile {xy} zoom {zoom2} from {repr(self)}'
                f' layer {repr(layer)} at {url}\n'
            )
        
        # Request the URL.
        response = requests.get(
            url, stream=True, headers=self.HTTP_REQUEST_HEADERS
        )
        
        # Read the image in the HTTP response content/body into a NumPy array of
        # pixel data.
        image = imageio.imread(response.content)
        
        return image
        
    # Grabs a 3x3 mosaic of tiles, with the specified tile at the centre.
    def tile_3x3(self, location=(0, 0), zoom=None, layer=None):
        (x, y), zoom2 = self.xy_zoom_from_location(location, zoom)
            
        # These offsets represent left, centre, and right, horizontally, and
        # top, middle, and bottom, vertically. The product of these gives us
        # our 3x3 mosaic.
        offsets = [-1, 0, 1]
        
        # Make image an empty NumPy array. We'll replace this when we know what
        # size and colour depth our tiles are going to be.
        image = np.array([])
        
        # For left, then centre, then right:
        for x_offset in offsets:
            # For top, then middle, then bottom (uh, I think…)
            for y_offset in offsets:
                # Grab the tile for this part of the mosaic.
                tile = self.tile(
                    location=(x + x_offset, y + y_offset), zoom=zoom2,
                    layer=layer
                )
                
                # If our image is the empty NumPy array (meaning we haven't
                # touched it yet), its size will be zero,
                # and this will be true:
                if not image.size:
                    # Now that we have an example, we can find out the height,
                    # width, and colour depth of our tiles:
                    tile_height, tile_width, n_channels = tile.shape
                    
                    # Make an array to fit 3 tiles by 3 tiles, matching the
                    # number of channels and the data type from the tile.
                    #
                    # For example, Google satellite imagery uses 256x256 pixel
                    # tiles with 3 channels (red, green, blue) per pixel, and
                    # the value for each channel is stored in an 8-bit unsigned
                    # integer (a whole number from 0-255). 8 bits of red, 8 bits
                    # of green, and 8 bits of blue gives us over 16 million
                    # possible colours.
                    image = np.zeros(
                        (tile_height * 3, tile_width * 3, n_channels),
                        dtype=tile.dtype
                    )
                    
                # x and y are the coordinates of the centre tile; x0 and y0 are
                # the coordinates of pixels in our mosaic.
                x0 = (x_offset + 1) * tile_width
                y0 = (y_offset + 1) * tile_height
                
                # Copy this tile's pixels into our mosaic at the right position.
                image[y0:y0+tile_height, x0:x0+tile_width] = tile
            
        return image

    def url(self, xy=(0, 0), zoom=0, layer=None):
        x, y = xy
        if self.uses_layers:
            if layer:
                # In __init__, we partially applied the format method to the
                # URL template, and have supplied the access token if
                # applicable. Get our finished URL by filling out the remaining
                # template variables, including the layer.
                url = self.url_partial(
                    layer=self.layer_key[layer], x=x, y=y, zoom=zoom
                )
            else:
                # The URL template needs a layer, but we didn't get one.
                raise ValueError(
                    f'{repr(self)} expects a layer, but none is given.'
                )
        else:
            # See above. Get our finished URL by filling out the remaining
            # template variables. This is for tile sources that don't use
            # layers.
            url = self.url_partial(x=x, y=y, zoom=zoom)

        return url
        
    # This overrides what we see when we inspect a TileSource in an interactive
    # Python session. For example, when we do this:
    #
    # >>> source = TileSource(name='google_maps', description, etc.)
    # >>> source
    #
    # Python will use the repr() function, which calls __repr__(), to determine
    # what to display. For example,
    #
    # <TileSource 'google_maps'>
    def __repr__(self):
        class_name = type(self).__name__
        return f'<{class_name} {repr(self.name)}>'