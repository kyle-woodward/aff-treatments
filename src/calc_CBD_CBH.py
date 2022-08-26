"""
Script used to calculate CBH and CBH for disturbed areas using
DIST, EVT, and (newly generated) CC and CH midpoint images
Usage:
    $ python calc_CBH_CBD.py -c path/to/config/file
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
    filename=os.path.join(os.path.dirname(__file__),'calc_CBH_CBD.log')
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
    #base_uri = "gs://landfire/LFTFCT_tables/{0}_Disturbance_Tbl.csv"
    base_uri2 = "gs://landfire/LFTFCT_tables/{0}_Disturbance_Tbl_filled.csv"

    # define the image collections for the raster data needed for calculations
    
    cc_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/cc")
    cbd_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/cbd")
    cbh_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/cbh")
    ch_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/ch")
    evt_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fvt")
    
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

    # cbd_img = ee.Image(
    #     cbd_ic.filter(ee.Filter.eq("version", 200))
    #     .limit(1, "system:time_start")
    #     .first()
    # )
    cbd_img = ee.Image("projects/pyregence-ee/assets/conus/fuels/Fuels_CBD_2021_12") # using FFv1 as baseline
    # Canopy base height image
    # cbh_img = ee.Image(
    #     cbh_ic.filter(ee.Filter.eq("version", 200))
    #     .limit(1, "system:time_start")
    #     .first()
    # )
    cbh_img = ee.Image("projects/pyregence-ee/assets/conus/fuels/Fuels_CBH_2021_12") # using FFv1 as baseline
    # EVT image
    evt_img = ee.Image(
        evt_ic.filter(ee.Filter.eq("version", 200))
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

    # to mask regression outputs for post-processing
    dist_mask = dist_img.mask() # this creates 1's everywhere include outside disturbed areas. not using

    #canopy guide collection for post-processing ruleset
    # AFF - create cg collection path from the dist_img_path
    cg_path = dist_img_path.replace('treatment_scenarios','fuelscapes_scenarios') + '/canopy_guide_collection'
    canopy_guide = ee.ImageCollection(f"{cg_path}").select('newCanopy').mosaic()
    
    # Here we are using the newly generated CC and CH as the midpoint images instead of FVH/C_Midpoint images
    # CC and CH are already binned to midpoint values during their calculation, only need to divide CH by 10 to get unscaled midpoint
    # Post-Disturbance Cover midpoint 
    post_cover_mid_img = ee.Image(dist_img_path.replace('treatment_scenarios','fuelscapes_scenarios') + '/CC')
    
    # Post-Disturbance Height midpoint 
    new_ch = ee.Image(dist_img_path.replace('treatment_scenarios','fuelscapes_scenarios') + '/CH')
    post_height_mid_img = new_ch.divide(10)
    
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

    # CBH #######################################################################################
    # define the collection to dump data to
    # each output will be an individual image so can be folder
    output_folder = out_folder_path

    uri = base_uri2.format("CBH")
    
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
    cbh = (
        intercept.expression(
            "b+(m1*x1)+(m2*x2)",
            {
                "b": intercept,
                "m1": hgt_scale,
                "x1": post_height_mid_img,
                "m2": cc_scale,
                "x2": post_cover_mid_img,
            }
        )
        .updateMask(dist_img)
        .multiply(10) #scale decimal regress output
        .toInt16()
        .clamp(0,100)
        .unmask(cbh_img) # fill un-disturbed pixels with fuel value
        .where(canopy_guide.eq(0), 0) # 0 where CG is 0
        .where(canopy_guide.eq(2), 100) # 100 (10m) where CG is 2
        .where(cc_img.eq(0), 0) # 0 where CG is 0
        .updateMask(zone_img)
        .rename('CBH')
        )
    cbh = cbh.where(cbh.gt(new_ch), new_ch.multiply(0.7).toInt16()).rename('CBH') # CBH can't be larger than CH; where it is, reduce CBH to 2/3 of CH
    
    # define where to export image
    output_asset = f"{output_folder}/CBH"

    # set up export task
    # export has specific CONUS projection/spatial extent
    task = ee.batch.Export.image.toAsset(
        image=cbh,
        description=f"export_CBH_{os.path.basename(dist_img_path)}",
        assetId=output_asset,
        region=cc_img.geometry(),
        crsTransform=geo_t,
        crs=crs,
        maxPixels=1e12,
    )
    task.start()  # kick of export task
    logger.info(f"Exporting {output_asset}")
    # logger.info(f"would export {output_asset}")

    # CBD #########################################################################################
    # get coefficients if pinion/juniper by conditional equation
    # 2017, 2019, 2025, 2059, 2115, 2116, 2119 values are 1
    pj = evt_img.expression(
        "(b('FVT') == 2017) ? 1 : "
        + "(b('FVT') == 2019) ? 1 : "
        + "(b('FVT') == 2025) ? 1 : "
        + "(b('FVT') == 2059) ? 1 : "
        + "(b('FVT') == 2115) ? 1 : "
        + "(b('FVT') == 2116) ? 1 : "
        + "(b('FVT') == 2119) ? 1 : 0"
    )

    # NOTES: here we use the original Canopy Height layer for coefficients calulation
    #   should this be the FVH midpoint or FVH image???
    # image -> band: ch_img -> CH
    # image -> band: fvh_img ->
    # image -> band: fvh_mid_img -> FVH_MIDPOINT
    
    ## this has been solved ^ we are using the newly calculated CH layer/10 which is already binned to midpoints

    # calculate the sh coefficients based on canopy height
    # uses conditional equation to check different levels
    sh1 = post_height_mid_img.expression(
        "(b('height') < 15) ? 0 : (b('height') < 30) ? 1 : 0"
    )
    sh2 = post_height_mid_img.expression(
        "(b('height') < 15) ? 0 : (b('height')) < 30 ? 0 : 1"
    )

    # apply CBD equation
    cbd = (
        ee.Image()
        .expression(
            " e ** (-2.4887057 + (0.0335917 * cov) + "
            "(-0.356861 * sh1) + -(0.6006381 * sh2) + "
            "(-1.10691 * pj) + (-0.0010804 * (cov * sh1)) "
            "+ (-0.0018324 * (cov * sh2)))",
            {
                "e": ee.Image.constant(math.e),
                "cov": post_cover_mid_img,
                "pj": pj,
                "sh1": sh1,
                "sh2": sh2,
            },
        )
        .updateMask(dist_img) 
        .multiply(100)
        .clamp(0,45)
        .toInt16()
        .unmask(cbd_img) # fill un-disturbed pixels with pre- fuel value
        .where(canopy_guide.eq(0), 0) # 0 where CG is 0
        .where(canopy_guide.eq(2), 1) # 1 (0.012kg/m^3) where CG is 2
        .where(canopy_guide.eq(3), 1) # 1 (0.012kg/m^3) where CG is 3
        .where(cc_img.eq(0), 0) # 0 where CC 2019 is 0
        .updateMask(zone_img)
        .rename("CBD")
    )
    
    # define where to export image
    output_asset = f"{output_folder}/CBD"

    # set up export task
    # export has specific CONUS projection/spatial extent (same as other images)
    task = ee.batch.Export.image.toAsset(
        image=cbd,
        description=f"export_CBD_{os.path.basename(dist_img_path)}",
        assetId=output_asset,
        region=cc_img.geometry(),
        crsTransform=geo_t,
        crs=crs,
        maxPixels=1e12,
    )
    task.start()  # kick off export
    logger.info(f"Exporting {output_asset}")
    # logger.info(f"would export {output_asset}")
# main level process if running as script
if __name__ == "__main__":
    main()