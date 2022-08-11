"""
Script used to calculate new Canopy Guide imgCollection for disturbed area using
DIST, BPS, EVH, EVC, and EVT images
Usage:
    $ python create_canopy_guide.py -c path/to/config
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
    filename=os.path.join(os.path.dirname(__file__), 'create_canopy_guide.log')
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
        description="CLI process for generating new canopy guide."
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
    # fbfm40_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fbfm40")
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
    
    old_cg = ee.ImageCollection("projects/pyregence-ee/assets/conus/fuels/canopy_guide_2021_12_v1").select('newCanopy').mosaic()
    # zone image to identify which pixel belong to zone
    zone_img = ee.Image("projects/pyregence-ee/assets/conus/landfire/zones_image")

    # define disturbance image used for the DIST codes
    # this will update with new disturbance info
    # can update with version tags of code
    dist_img = ee.Image(
        f"{dist_img_path}"
    ).unmask(0)

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
    # output_ic = f"projects/pyregence-ee/assets/conus/fuels/canopy_guide_{version}"
    output_ic = f"{out_folder_path}/canopy_guide_collection" # canopy guide is exported as zone-wise imgs into its own imageCollection, so we need to back up one path to the parent folder and make a canopy guide imgColl
    os.popen(f"earthengine create collection {output_ic}")
    
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
        to_codes = to_numeric(ee.List(table.get("NewCanopy")))

        # apply the remapping encoded values -> NewCanopy values
        zone_newcanopy_remapped = encoded_img.remap(from_codes, to_codes) #non-matches return null (masked) value

        dist_high_harvest = dist_img.selfMask().lte(333).bitwiseAnd(dist_img.selfMask().gte(331)) # high harvest = 1

        # Initialize a CG raster of 1's and burn in actual remapped CG values overtop (CG=1 means leave fuels value as-is)
        # then mask areas that are not current zone        
        zone_newcanopy = (           
            #ee.Image.constant(1) 
            old_cg # AFF - starting with FFv1 canopy guide, not making CG from scratch
            .where(dist_img.selfMask(), zone_newcanopy_remapped) #returns 1 if zone_newcanopy_remapped is null in disturbed area
            .where(dist_high_harvest.eq(1),0) # zero out CG in high harvest disturbed areas
            .updateMask(zone_img.eq(zone))
            .rename("newCanopy")
            .byte() #valid values are 0-3
        )
                
        # create an image with information of what happened where
        # if disturbed and has new value flag = 0
        # if not distubed (i.e. initial 1 value) flag = 1
        # if disturbed and has no remapped code flag = 2
        # if outside of zone flag = 3
        flags = (
            dist_img.Not() 
            .where(zone_newcanopy.add(1).selfMask().eq(0), 2) # .add(1) so 0 is no longer a valid value and can be masked
            .where(zone_img.neq(zone), 3)
            .updateMask(zone_img.selfMask())
            .uint8()
            .rename("qa_flags")
        )

        # combine new CG layer and flags
        zone_out = ee.Image.cat([zone_newcanopy, flags]).set(
            "zone", zone
        )  # set zone metadata

        # set up export task
        # each zone will be all of CONUS with same projection/spatial extent
        # this is to prevent any pixel misalignment at edges of zone
        asset_id = output_ic + f"/new_canopy_zone{zone:02d}"
        task = ee.batch.Export.image.toAsset(
            image=zone_out,
            description=f"Zone{zone:02d}_canopy_guide_export_{os.path.basename(dist_img_path)}",
            assetId=asset_id,
            region=dist_img.geometry(),
            scale=scale,
            crs=crs,
            maxPixels=1e12,
            pyramidingPolicy={".default": "mode"},
        )
        logger.info(f"Exporting {asset_id}")
        task.start()
    
        
# main level process if running as script
if __name__ == "__main__":
    main()