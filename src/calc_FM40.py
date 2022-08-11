"""
Script used to calculate new FM40 values for disturbed area using
DIST, BPS, FVH, FVC, and FVT images
Usage:
    $ python calc_fm40.py -c path/to/config
"""
import os 
import ee
import yaml
import argparse
import logging
from utils.ee_csv_parser import parse_txt, to_numeric

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.WARNING,
    filename=os.path.join(os.path.dirname(__file__), 'calc_fm40.log')
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ee.Initialize()

# this function is currently not used because the encoded values are precomputed in cmb_table_qa
# will keep just in case...
def encode_table(table: ee.Dictionary):
    """Function to take dictionary representation of CSV and
    encode the DIST, BPS, EVH, EVC, and EVT values into a unique code
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
        evt = ee.Number.parse(evtr.get(i))
        dist = ee.Number.parse(distr.get(i))
        evc = ee.Number.parse(evcr.get(i))
        evh = ee.Number.parse(evhr.get(i))
        bps = ee.Number.parse(bpsrf.get(i))

        # encoding equation
        new_code = ee.Number.expression(
            "a*as+ b*bs + c*cs + d*ds + e*es",
            {
                "a": dist,
                "as": 1e13,
                "b": bps,
                "bs": 1e10,
                "c": evh,
                "cs": 1e7,
                "d": evc,
                "ds": 1e4,
                "e": evt,
                "es": 1e0,
            },
        )

        return new_code

    # parse out the individual columns as lists
    # used in `combine` to extract values by index
    evtr = ee.List(table.get(evtr_name))
    distr = ee.List(table.get(dist_name))
    evcr = ee.List(table.get(evcr_name))
    evhr = ee.List(table.get(evhr_name))
    bpsrf = ee.List(table.get(bpsrf_name))

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
    
    
    # define where the cmb tables can be found on cloud storage
    # these need to be the preprocessed tables from cmb_table_qa
    base_uri = "gs://landfire/LFTFCT_tables/cmb_zones_wneighbors/z{:02d}_CMB.csv"

    # define the column information used to for encode function
    evtr_name = "EVTR"
    dist_name = "DIST"
    evcr_name = "EVCR"
    evhr_name = "EVHR"
    bpsrf_name = "BPSRF"

    # define a list of zone information
    # does a skip from 67 to 98...not sure why just the zone numbers
    #zones = list(range(1, 67)) + [98, 99] # all CONUS zones used for FireFactor.. check which zones your AOI falls in and provide them as a list
    zones = [6] # For AFF project, entire AOI falls in LF Zone 6

    # define the image collections for the raster data needed for calculations
    bps_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/bps")
    fbfm40_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fbfm40")
    # the actual values being used in the FM40 crosswalk are the FVH, FVC, FVT
    # however, the tables have EVH, EVC, EVT...
    # so variables are named as in the tables but note they are actually the F* layers
    evt_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fvt")
    evh_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fvh")
    evc_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fvc")

    # start by extracting out the specific image we need for the FM40 calculation
    # we need the version 200 / year 2016 data
    # sometimes the date metadata is not actually 2016 so we filter by version as select first image in time
    # BPS image
    bps_img = ee.Image(
        bps_ic.filter(ee.Filter.eq("version", 200))
        .limit(1, "system:time_start")
        .first()
    )
    # EVT image
    evt_img = ee.Image(
        evt_ic.filter(ee.Filter.eq("version", 200))
        .limit(1, "system:time_start")
        .first()
    )
    # EVH image
    evh_img = ee.Image(
        evh_ic.filter(ee.Filter.eq("version", 200))
        .limit(1, "system:time_start")
        .first()
    )
    # EVC image
    evc_img = ee.Image(
        evc_ic.filter(ee.Filter.eq("version", 200))
        .limit(1, "system:time_start")
        .first()
    )
    # FM40 image from previous time, used when there is no distubance 
    # oldfm40_img = ee.Image(
    #     fbfm40_ic.filter(ee.Filter.eq("version", 200))
    #     .limit(1, "system:time_start")
    #     .first()
    # )
    # FOR AFF do we use FireFactor v1 updated FM40 and update in only AFF disturbed pixels? 
    # or do we update from LF2016 2019 capable like we start with for FireFactor and combine FF v1 DIST img w AFF DIST img?
    # if we do second option, then we won't have WUI nonburnable conversions in there, but if we do first option, the most important DIST from the last 10 years will take precedent at each pixel
    oldfm40_img = ee.ImageCollection("projects/pyregence-ee/assets/conus/fuels/Fuels_FM40_collection_fix_bug_052022").select('new_fbfm40').mosaic()
    
    # zone image to identify which pixel belong to zone
    zone_img = ee.Image("projects/pyregence-ee/assets/conus/landfire/zones_image")

    # define disturbance image used for the DIST codes
    # this will update with new disturbance info
    # can update with version tags of code
    dist_img = ee.Image(
        f"{dist_img_path}"
    )#.unmask(0) # to ensure encoded imgs that get remapped to new FM40 lookup values only occur in the original masked DIST img pixels
    
    
    # encode the images into unique codes
    # code will be a 16 digit value where each group of values
    # are the individual values from the images
    encoded_img = dist_img.expression(
        "a*as+ b*bs + c*cs + d*ds + e*es",
        {
            "a": dist_img,
            "as": 1e13,
            "b": bps_img,
            "bs": 1e10,
            "c": evh_img,
            "cs": 1e7,
            "d": evc_img,
            "ds": 1e4,
            "e": evt_img,
            "es": 1e0,
        },
    )

    # define the collection to dump data to
    # this needs to be an image collection as each zone is exported individually
    output_ic = f"{out_folder_path}/fm40_collection" # canopy guide is exported as zone-wise imgs into its own imageCollection, so we need to back up one path to the parent folder and make a canopy guide imgColl
    os.popen(f"earthengine create collection {output_ic}").read()
    
    # loop through each zone to do the FM40 calculation
    for zone in zones:
        # skip over zone 11, there is no zone 11
        if zone == 11:
            continue

        # plug in the zone value into the table uri string
        uri = base_uri.format(zone)
        # read in the table from cloud storage
        blob = ee.Blob(uri)

        # parse the table as an ee.Dictionary
        table = parse_txt(blob)

        # legacy code to encode the table values if not done so already
        # from_codes = ee.List(encode_table(table))

        # read in the encoded value list as numeric
        from_codes = to_numeric(ee.List(table.get("encoded")))
        # read the list of values to remap to as numeric
        to_codes = to_numeric(ee.List(table.get("NewFBFM40")))

        # apply the remapping encoded values -> new FM40 values
        zone_fm40_remapped = encoded_img.remap(from_codes, to_codes) 
        
        # replace all values in old fm40 raster that are disturbed with new fm40 values
        # then mask areas that are not current zone
        zone_fm40 = (
            oldfm40_img.where(dist_img.selfMask(), zone_fm40_remapped) # .where(dist_img.selfMask(), zone_fm40_remapped) returns input value if test value is false, i.e. if no 
            .updateMask(zone_img.eq(zone))
            .rename("new_fbfm40")
            .uint16()
        )

        # create an image with information of what happened where
        # if disturbed and has new FM40 value flag = 0
        # if not distubed (ie old FM40 value) flag = 1
        # if disturbed and new FM40 has no remapped code flag = 2
        # if outside of zone flag = 4
        flags = (
            dist_img.Not()
            .where(zone_fm40.selfMask().eq(0), 2)
            .where(zone_img.neq(zone), 3)
            .updateMask(zone_img.selfMask())
            .uint8()
            .rename("qa_flags")
        )

        # combine new FM40 layer and flags
        zone_out = ee.Image.cat([zone_fm40, flags,]).set(
            "zone", zone
        )  # set zone metadata

        # set up export task
        # each zone will be all of CONUS with same projection/spatial extent
        # this is to prevent any pixel misalignment at edges of zone
        asset_id = output_ic + f"/FM40_zone{zone:02d}"
        task = ee.batch.Export.image.toAsset(
            image=zone_out.clip(dist_img.geometry()), #clip to just the AFF study area bbox
            description=f"Zone{zone:02d}_FM40_export_{os.path.basename(dist_img_path)}",
            assetId=asset_id,
            region=zone_img.geometry(),
            #crsTransform=geo_t, 
            scale=scale,
            crs=crs, 
            maxPixels=1e12,
            pyramidingPolicy={".default": "mode"},
        )
        logger.info(f"Exporting {asset_id}")
        task.start()  # kick of export task
    

# main level process if running as script
if __name__ == "__main__":
    main()