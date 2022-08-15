"""
Script used to calculate CC and CH for disturbed areas using
DIST, EVT, and FVC/FVH midpoint images
Usage:
    $ python calc_CC_CH.py -c path/to/config/file
"""
import os
import ee
import math
import yaml
import argparse
import logging
from utils.ee_csv_parser import parse_txt, to_numeric

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.WARNING,
    filename=os.path.join(os.path.dirname(__file__),'calc_CC_CH.log')
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ee.Initialize()

def encode_table(table):
    """Function to take dictionary representation of CSV and
    encode the DIST and EVT values into a unique code
    args:
        table (ee.Dictionary): dictionary representation of csv table
    returns:
        ee.List: list of unique encoded values
    """

    def combine(i):
        """closure function to do the encoding per row"""
        # set row index to number
        i = ee.Number(i)
        # parse out the individual columns as numbers
        evt = ee.Number(evtr.get(i))
        dist = ee.Number(distr.get(i))

        # encoding equation
        new_code = ee.Number.expression(
            "a*as + b*bs",
            {
                "a": dist,
                "as": 1e4,
                "b": evt,
                "bs": 1e0,
            },
        )

        return new_code

    # define the column information used to for encode function
    evtr_name = "EVT_Fill"
    dist_name = "HDist"

    # parse out the individual columns as lists
    # used in `combine` to extract values by index
    evtr = to_numeric(ee.List(table.get(evtr_name)))
    distr = to_numeric(ee.List(table.get(dist_name)))

    # get number of rows to loop over
    n = evtr.length()

    # run the encoding process on each row
    encoded = ee.List.sequence(0, n.subtract(1)).map(combine)

    return encoded

def main():
    """Main level function for generating new CBH and CBD"""
    
    # initalize new cli parser
    parser = argparse.ArgumentParser(
        description="CLI process for generating new CBH and CBD."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to config file",
    )

    parser.add_argument(
        "-d",
        "--dist_img_path",
        type=str,
        help="asset path of input DIST img"

    )
    
    parser.add_argument(
        "-o",
        "--out_folder_path",
        type=str,
        help="asset path of output folder"

    )
    args = parser.parse_args()

    dist_img_path = args.dist_img_path
    out_folder_path = args.out_folder_path

    # parse config file
    with open(args.config) as file:
        config = yaml.full_load(file)

    geo_info = config["geo"]
    #version = config["version"].get('latest')

    # extract out geo information from config
    geo_t = geo_info["crsTransform"]
    scale = geo_info["scale"]
    x_size, y_size = geo_info["dimensions"]
    crs = geo_info["crs"]


    # define where the distrubance regression tables can be found on cloud storage
    # defining two because CBH was preprocess and QA'ed so points to updated table
    base_uri = "gs://landfire/LFTFCT_tables/{0}_Disturbance_Tbl.csv"
    base_uri2 = "gs://landfire/LFTFCT_tables/{0}_Disturbance_Tbl_filled.csv"

    # define the image collections for the raster data needed for calculations
    cc_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/cc")
    cbh_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/cbh")
    ch_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/ch")
    evt_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fvt")
    fvc_mid_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/fuels/Midpoint_CC")
    fvh_mid_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/fuels/Midpoint_CH")

    # start by extracting out the specific image we need for the regression calculations
    # we need the version 200 / year 2016 data
    # sometimes the date metadata is not actually 2016 so we filter by version as select first image in time
    # Canopy cover image
    # cc_img = ee.Image(
    #     cc_ic.filter(ee.Filter.eq("version", 200)).limit(1, "system:time_start").first()
    # )
    # AFF - we use FFv1 layers as baseline, updating only in DIST img areas
    cc_img = ee.Image("projects/pyregence-ee/assets/conus/fuels/Fuels_CC_2021_12") # using FFv1 as baseline
    # Canopy height image
    # ch_img = ee.Image(
    #     ch_ic.filter(ee.Filter.eq("version", 200)).limit(1, "system:time_start").first()
    # )
    ch_img = ee.Image("projects/pyregence-ee/assets/conus/fuels/Fuels_CH_2021_12") # using FFv1 as baseline
    # Canopy base height image
    cbh_img = ee.Image(
        cbh_ic.filter(ee.Filter.eq("version", 200))
        .limit(1, "system:time_start")
        .first()
    )
    # EVT image
    evt_img = ee.Image(
        evt_ic.filter(ee.Filter.eq("version", 200))
        .limit(1, "system:time_start")
        .first()
    )
    # FVC midpoint image
    fvc_mid_img = ee.Image(
        fvc_mid_ic.filter(ee.Filter.eq("version", 200))
        .limit(1, "system:time_start")
        .first()
    )
    # FVH midpoint image
    fvh_mid_img = ee.Image(
        fvh_mid_ic.filter(ee.Filter.eq("version", 200))
        .limit(1, "system:time_start")
        .first()
    )
    # zone image to identify which pixel belong to zone
    zone_img = ee.Image("projects/pyregence-ee/assets/conus/landfire/zones_image")

    # define disturbance image used for the DIST codes
    # this will update with new disturbance info
    # can update with version tags of code
    dist_img = ee.Image(
        f"{dist_img_path}"
    )

    # get binary image of where disturbance happened
    dist_mask = dist_img.mask() # this creates 1's everywhere include outside disturbed areas. not using

    #canopy guide collection for post-processing ruleset
    # AFF - create cg collection path from the dist_img_path
    cg_path = dist_img_path.replace('treatment_scenarios','fuelscapes_scenarios') + '/canopy_guide_collection'
    canopy_guide = ee.ImageCollection(f"{cg_path}").select('newCanopy').mosaic()
    
    # encode the images into unique codes
    # code will be a 7 digit value where each group of values
    # are the individual values from the images
    encoded_img = dist_img.expression(
        "a*as + b*bs",
        {
            "a": dist_img,
            "as": 1e4,
            "b": evt_img,
            "bs": 1e0,
        },
    )

    # define the variable to process in loop
    # these all have the same process for regression
    vars = ["Cover", "Height"]  

    # get the imagery to use when not disturbed
    # must match serquence of vars
    fills = [cc_img, ch_img]

    # output name for export
    output_names = ['CC', 'CH']
    
    # define the collection to dump data to
    # each output will be an individual image so can be folder
    output_folder = out_folder_path

    # loop through the variables to run the regressions
    for i, var in enumerate(vars):                    
        # if i==2 or var == CBH use the base_uri2
        # this was to accommodate CBH, which was removed from vars list so the else condition will be true for both now
        if i == 2:
            uri = base_uri2.format(var)
        else:
            uri = base_uri.format(var)

        # read in the table from cloud storage
        blob = ee.Blob(uri)

        # parse the table as an ee.Dictionary
        table = parse_txt(blob)

        # apply encoding process to table (in-memory)
        from_codes = ee.List(encode_table(table))

        # extract out the individual coefficients for the EVT/DIST combinations
        intercept_codes = to_numeric(ee.List(table.get("intercept")))
        hgt_scale_codes = to_numeric(ee.List(table.get("HT_coef")))
        cc_scale_codes = to_numeric(ee.List(table.get("CC_coef")))

        # apply remapping process to get images of coefficients for regression
        intercept = encoded_img.remap(from_codes, intercept_codes)
        hgt_scale = encoded_img.remap(from_codes, hgt_scale_codes)
        cc_scale = encoded_img.remap(from_codes, cc_scale_codes)

        # apply the regression equation for the variable
        # and fill in areas not disturbed with original variable image
        regress = (
            intercept.expression(
                "b+(m1*x1)+(m2*x2)",
                {
                    "b": intercept,
                    "m1": hgt_scale,
                    "x1": fvh_mid_img,
                    "m2": cc_scale,
                    "x2": fvc_mid_img,
                }
            )
            
        )
        
        if var == 'Cover': # CC 
            
            values = ee.List.sequence(0,100)
            bins = ee.List.repeat(0,10)\
                .cat(ee.List.repeat(15,10))\
                .cat(ee.List.repeat(25,10))\
                .cat(ee.List.repeat(35,10))\
                .cat(ee.List.repeat(45,10))\
                .cat(ee.List.repeat(55,10))\
                .cat(ee.List.repeat(65,10))\
                .cat(ee.List.repeat(75,10))\
                .cat(ee.List.repeat(85,10))\
                .cat(ee.List.repeat(95,11))
            
            regress_processed = (
                regress
                .updateMask(dist_img)
                .clamp(0,100)
                .toInt16()
                .remap(values,bins,0)
                .unmask(fills[i])
                .where(canopy_guide.eq(0), 0) # zero out where CG is 0
                .where(cc_img.eq(0), 0) # zero out where CC 2019 is 0
                .updateMask(zone_img)
                .rename(var.lower())
                )
        
        else: # CH 
            
            values = ee.List.sequence(0,51)
            bins = ee.List.repeat(0,1)\
                .cat(ee.List.repeat(3,4))\
                .cat(ee.List.repeat(7,4))\
                .cat(ee.List.repeat(11,4))\
                .cat(ee.List.repeat(15,4))\
                .cat(ee.List.repeat(19,4))\
                .cat(ee.List.repeat(23,4))\
                .cat(ee.List.repeat(27,4))\
                .cat(ee.List.repeat(31,4))\
                .cat(ee.List.repeat(35,4))\
                .cat(ee.List.repeat(39,4))\
                .cat(ee.List.repeat(43,4))\
                .cat(ee.List.repeat(47,4))\
                .cat(ee.List.repeat(51,3))

            regress_processed = (
                regress
                .updateMask(dist_img)
                .toInt16()
                .remap(values,bins,0)
                .multiply(10)
                .clamp(0,510)
                .unmask(fills[i])
                .where(canopy_guide.eq(0), 0) # zero out where CG is 0
                .where(cc_img.eq(0), 0) # zero out where CC is 0
                .updateMask(zone_img)
                .rename(var.lower()) 
                )
        
        # define where to export image
        output_asset = f"{output_folder}/{output_names[i]}"

        # set up export task
        # export has specific CONUS projection/spatial extent
        task = ee.batch.Export.image.toAsset(
            image=regress_processed,
            description=f"export_{output_names[i]}_{os.path.basename(dist_img_path)}",
            assetId=output_asset,
            region=cc_img.geometry(),
            crsTransform=geo_t,
            crs=crs,
            maxPixels=1e12,
        )
        task.start()  # kick off export task
        logger.info(f"Exporting {output_asset}")
        # logger.info(f"CG path: {cg_path}")
        # logger.info(f"would export {output_asset}")
# main level process if running as script
if __name__ == "__main__":
    main()