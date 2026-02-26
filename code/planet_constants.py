"""
Filename: planet_constants.py
Author: Michele Lissoni
Date: 2026-02-16
"""

"""

Constants for the planet Mercury. Need to be changed if a different body is considered.

"""

from pyproj import Transformer, CRS, Geod

class Planet_Properties:

    RADIUS = 2439400
    ES = 0
    
# CRS Well-Known Text representations for the latlon CRS, the Equidistant Cylindrical CRS and the Cylindrical Equal-Area CRS
    
class Planet_CRS_WKT:

    LATLON_WKT = f'''
        GEOGCRS["Mercury_2015",
            DATUM["Mercury_2015",
                ELLIPSOID["Mercury_2015",
                    {Planet_Properties.RADIUS},
                    {Planet_Properties.ES},
                    LENGTHUNIT["metre",1]
                ]
            ],
            PRIMEM["Reference_Meridian",0,ANGLEUNIT["degree",0.0174532925199433]],
            CS[ellipsoidal,2],
            AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],
            AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],
            USAGE[SCOPE["Not known."],
            AREA["World."],
            BBOX[-90,-180,90,180]],
            ID["ESRI",104974]
        ]
        '''        

    CYL_WKT = f'''
        PROJCRS["SimpleCylindrical Mercury",
            BASEGEOGCRS["GCS_Mercury",
                DATUM["D_Mercury",
                    ELLIPSOID["Mercury",
                        {Planet_Properties.RADIUS},
                        {Planet_Properties.ES},
                        LENGTHUNIT["metre",1,ID["EPSG",9001]]
                    ]
                ],
                PRIMEM["Reference_Meridian",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]
            ],
            CONVERSION["Equidistant Cylindrical",
                METHOD["Equidistant Cylindrical",ID["EPSG",1028]],
                PARAMETER["Latitude of 1st standard parallel",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8823]],
                PARAMETER["Longitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],
                PARAMETER["False easting",0,LENGTHUNIT["metre",1],ID["EPSG",8806]],
                PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]
            ],
            CS[Cartesian,2],
            AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],
            AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]
        ]
        '''
        
    CEA_WKT = f'''
        PROJCRS["Sphere_Cylindrical_Equal_Area",
            BASEGEOGCRS["GCS_Mercury",
                DATUM["D_Mercury",
                    ELLIPSOID["Mercury",
                        {Planet_Properties.RADIUS},
                        {Planet_Properties.ES},
                        LENGTHUNIT["metre",1,ID["EPSG",9001]]
                    ]
                ],
                PRIMEM["Reference_Meridian",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]
            ],
            CONVERSION["Sphere_Cylindrical_Equal_Area",
                METHOD["Lambert Cylindrical Equal Area (Spherical)",ID["EPSG",9834]],
                PARAMETER["Latitude of 1st standard parallel",0,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8823]],
                PARAMETER["Longitude of natural origin",0,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8802]],
                PARAMETER["False easting",0,LENGTHUNIT["metre",1],ID["EPSG",8806]],
                PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]
            ],
            CS[Cartesian,2],
            AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],
            AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]
        ]
        '''

# Pyproj CRS objects for the above CRS

class Planet_CRS:

    LATLON_CRS = CRS.from_wkt(Planet_CRS_WKT.LATLON_WKT)
    
    CYL_CRS = CRS.from_wkt(Planet_CRS_WKT.CYL_WKT)
    
    CEA_CRS = CRS.from_wkt(Planet_CRS_WKT.CEA_WKT)
    
# Parameters of the basemap mosaic
    
class Planet_Basemap:

    BASEMAP_RESOLUTION = 665.24315271
    
    BASEMAP_HEIGHT = 11520
    BASEMAP_WIDTH = 23040
    
    BASEMAP_EXTENT = ((-7663601.1191670000553131, 7663601.1192713994532824),(-3831800.5596361998468637, 3831800.5595829999074340))
    
    BASEMAP_CRS = Planet_CRS.CYL_CRS
    
PLANET_GEOD = Geod(a = Planet_Properties.RADIUS, es = Planet_Properties.ES) # Pyproj.Geod to compute great-circle distances and more

# Function that returns a Pyproj.CRS of an Azimuthal Equidistant CRS centered on given coordinates

def getAzCRS(center_lon, center_lat):

    az_wkt = f'''
        PROJCRS["Custom AzimuthalEquidistant Mercury",
            BASEGEOGCRS["GCS_Mercury",
                DATUM["D_Mercury",
                    ELLIPSOID["Mercury",
                        {Planet_Properties.RADIUS},
                        {Planet_Properties.ES},
                        LENGTHUNIT["metre",1,ID["EPSG",9001]]
                    ]
                ],
                PRIMEM["Reference_Meridian",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]
            ],
            CONVERSION["unknown",
                METHOD["Modified Azimuthal Equidistant",ID["EPSG",9832]],
                PARAMETER["Latitude of natural origin",
                    {center_lat},
                    ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]
                ],
                PARAMETER["Longitude of natural origin",
                    {center_lon},
                    ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]
                ],
                PARAMETER["False easting",0,LENGTHUNIT["metre",1],ID["EPSG",8806]],
                PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]
            ],
            CS[Cartesian,2],
            AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],
            AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]
        ]
        '''
        
    az_crs = CRS.from_wkt(az_wkt)
    
    return az_crs

