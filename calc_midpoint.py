"""
Script used to calculate FVH and FVC midpoint images for inputs to calc_CC_CH.py
NOTE: If new FVC and FVH version is released by landfire, will need to upload those imgs to the fvc and fvh imagecollections 
and re-run this script
technical notes: 
this batch exports imgs to pre-existing imgCollections, so export tasks for assets that already exist will fail
(which is ok since those do not need to be updated) 
if you want to avoid seeing failed export tasks when running, can implement
an 'earthengine ls' call first to collect pre-existing img asset names in the collection and skip them in the batch process
Currently don't see need in config-based versioning the FVC and FVH_Midpoint imgCollections as 
they will hold the newer midpoint layers when updates are necessary
Usage:
    $ python calc_midpoint.py
"""

import ee
from utils.ee_csv_parser import parse_txt, to_numeric

ee.Initialize()

def batch_export(ic: ee.ImageCollection, output_ic: str):
    """Function to loop through each image in ImageCollection and export
    args:
        ic (ee.ImageCollection): image collection to export
        output_ic (str): name of image collection to export images to
    returns:
        None
    """
    # get number of images in collection
    n = ic.size()

    # get the image collection as a list to look up images using index
    ic_list = ic.toList(n)

    # region definition for test exports
    # test_region = ee.Geometry.Rectangle([-124.34413273497427,44.216470686379765,-116.91737492247427,46.96788804818362])

    # loop over all of the images
    for i in range(n.getInfo()):
        # get image from index
        img = ee.Image(ic_list.get(i))
        # extract export region
        export_region = img.geometry()

        # get the orignal image name info
        img_name = img.get("system:index").getInfo()
        name_comp = img_name.split("_")

        # throw the word MIDPOINT into the mix
        out_name = "_".join(name_comp[:2] + ["MIDPOINT"] + name_comp[-1:])
        # give the export a name
        descr = f"{out_name.replace('_','')} Export"
        print(f"Running export for {out_name}")

        # set up export task for image
        task = ee.batch.Export.image.toAsset(
            img,
            description=descr,
            assetId=f"{output_ic}/{out_name}",
            pyramidingPolicy={".default": "mode"},
            region=export_region,
            scale=30,
            maxPixels=1e12,
        )
        task.start()  # kick off export

    return


def main():
    """Main level function for executing the new FM40 calculations"""

    # define closure functions to do the remapping
    # needs to be in the main namespace so we can use the lookup tables in the functions as well as map over ee.ImageCollections
    def evc_midpoint(img):
        """closure function to apply remapping of canopy cover values to midpoint values"""
        # read in the columns from look up dictionarys as numeric values
        evc_cls = to_numeric(ee.List(fvc_lut.get(evc_col)))
        midpoints = to_numeric(ee.List(fvc_lut.get(mid_col)))

        # apply remapping process and copy over metadata
        return (
            img.remap(evc_cls, midpoints)
            .rename("FVC_MIDPOINT")
            .copyProperties(img, ["system:start_time", "system:end_time", "version"])
        )

    def evh_midpoint(img):
        """closure function to apply remapping of canopy height values to midpoint values"""
        # read in the columns from look up dictionarys as numeric values
        evh_cls = to_numeric(ee.List(fvh_lut.get(evh_col)))
        midpoints = to_numeric(ee.List(fvh_lut.get(mid_col)))

        # apply remapping process and copy over metadata
        return (
            img.remap(evh_cls, midpoints)
            .rename("FVH_MIDPOINT")
            .float()
            .copyProperties(img, ["system:start_time", "system:end_time", "version"])
        )

    # define the image collections for FVC and FVH
    fvc_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fvc")
    fvh_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fvh")

    # read in the tables with the midpoint reclassification values
    fvc_blob = ee.Blob("gs://landfire/LFTFCT_tables/LUT_Cover.csv")
    fvh_blob = ee.Blob("gs://landfire/LFTFCT_tables/LUT_Height_trimmed.csv")

    # parse the CSV tables as ee.Dictionary
    fvc_lut = parse_txt(fvc_blob)
    fvh_lut = parse_txt(fvh_blob)

    # define the column information used to for remapping functions
    evc_col = "EVC"
    evh_col = "EVH"
    mid_col = "MidPoint"

    # apply the remapping process to each image in the FVC/FVH collections
    fvc_mid_ic = fvc_ic.map(evc_midpoint)
    fvh_mid_ic = fvh_ic.map(evh_midpoint)

    # define the output collection names
    fvc_mid_output = "projects/pyregence-ee/assets/conus/fuels/Midpoint_CC"
    fvh_mid_output = "projects/pyregence-ee/assets/conus/fuels/Midpoint_CH"

    # run the exports for the new FVC/FVH midpoint collections
    batch_export(fvc_mid_ic, fvc_mid_output)
    batch_export(fvh_mid_ic, fvh_mid_output)


# main level process if running as script
if __name__ == "__main__":
    main()