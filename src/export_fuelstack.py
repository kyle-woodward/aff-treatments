#%%
import os 
import yaml
import ee
import logging
import datetime
from src.utils.cloud_utils import poll_submitted_task

repo_dir =  os.path.abspath(os.path.join(__file__ ,"../.."))
date_id = datetime.datetime.utcnow().strftime("%Y-%m-%d").replace('-','') # like 20221216
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.WARNING,
    filename = os.path.join(repo_dir,"log",f"{date_id}.log")
)
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

try:
    credentials = ee.ServiceAccountCredentials(email=None,key_file='/home/private-key.json')
    ee.Initialize(credentials)
except:
    ee.Initialize(project="pyregence-ee")


def export_fuelstack(folder:str,poll=False):
    """
    exports fuelstack (CC,CH,CBH,CBD,FM40) into a mutliband cloud optimized geotiff on google cloud storage
    args:
        folder (str): 'fuelscape' folder on GEE
    returns:
        GCS path to exported COG
    """
    # Set a default workload tag.
    ee.data.setDefaultWorkloadTag('smtx-compute')
    
    # parse config file
    config_file = os.path.join(repo_dir,'config.yml')
    with open(config_file) as file:
        config = yaml.full_load(file)

    geo_info = config["geo"]

    # extract out geo information from config
    crs = geo_info["crs"]
    scale= geo_info["scale"]
    
    fm40 = ee.ImageCollection(f"{folder}/fm40_collection").select('new_fbfm40').mosaic()
    cc = ee.Image(f"{folder}/CC")
    ch = ee.Image(f"{folder}/CH")
    cbh = ee.Image(f"{folder}/CBH")
    cbd = ee.Image(f"{folder}/CBD")
    fuel_stack = fm40.addBands(cc).addBands(ch).addBands(cbh).addBands(cbd).rename('FM40','CC','CH', 'CBH','CBD').toInt16()
    logger.info(f"exported fuelstack bandnames: {fuel_stack.bandNames().getInfo()}")

    # export params
    bucket='sm-tx'
    scn_id = folder.split('/')[-1]
    desc = f"GCS export: {scn_id}"
    # fileNamePrefix=folder.split(f'{bucket}/')[-1]
    fileNamePrefix='/'.join(folder.split('/')[-3:])
    logger.info(fileNamePrefix)
    test_rect = ee.Geometry.Polygon([[-121.03140807080943,39.716216195378465],
                                    [-120.87347960401256,39.716216195378465],
                                    [-120.87347960401256,39.798563143043246],
                                    [-121.03140807080943,39.798563143043246],
                                    [-121.03140807080943,39.716216195378465]])
    # set workload tag context with with 
    with ee.data.workloadTagContext('smtx-export'):
        task = ee.batch.Export.image.toCloudStorage(
            image=fuel_stack,
            description=desc,
            bucket=bucket,
            fileNamePrefix=fileNamePrefix,
            region = ee.Image(cc).geometry(), #test_rect
            scale=scale,
            crs=crs)
        task.start()
        gcs_file = f"gs://{bucket}/{fileNamePrefix}.tif"
        logger.info(f'Export started: {gcs_file}')
        
    if poll:
        poll_submitted_task(task,1) 
    
    return gcs_file

#%%
#export_fuelstack(folder="projects/pyregence-ee/assets/sm-tx/test2/outputs/Treatments_Plumas_Protect_Alt1_fuelscape")
# %%
