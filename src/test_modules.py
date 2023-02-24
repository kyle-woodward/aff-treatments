#%%
import os
from utils.cloud_utils import download_gcs, ee_task_list_poller
from upload_treatments import upload_treatments
from rasterize_treatments import rasterize
from create_canopy_guide import cg
from calc_FM40 import fm40
from calc_CC_CH import cc_ch
from calc_CBD_CBH import cbd_cbh
from export_fuelstack import export_fuelstack
import time
# Per-Project File Structure
# GCS
# gs://op-tx
#       /project_folder
#         /inputs
#           **treatments.zip**
#           **aoi.zip**
#         /outputs
#           **fuelscape.tif**
# GEE
# projects/pyregence-ee/assets/op-tx
#                               /project_folder
#                                   /inputs
#                                      ** treatments **
#                                      ** aoi **
#                                   /outputs
#                                       /*_fuelscape
#                                           ** cg_collection/**
#                                           ** fm40_collection/ **
#                                           ** cc **
#                                           ** ch **
#                                           ** cbh **
#                                           ** cbd **
# input arguments to the functions
shapefile="C:\pc448\data\inputs\Treatments_Plumas_Protect_Alt1\Treatments_Plumas_Protect_Alt1.shp"
#%%
treatment_asset, aoi_asset = upload_treatments(project_name='test3',shapefile=shapefile)
print(treatment_asset,aoi_asset)
# poll for upload tasks
time.sleep(5)
ee_task_list_poller('Upload',2,5)
#%%
#whie loop polling...
## CHECKPOINT
treatment_asset="projects/pyregence-ee/assets/op-tx/test3/inputs/Treatments_Plumas_Protect_Alt1"
aoi_asset = "projects/pyregence-ee/assets/op-tx/test3/inputs/Treatments_Plumas_Protect_Alt1_AOI"
#%%
DIST_raster = rasterize(input=treatment_asset,rasterize_on='DIST',aoi=aoi_asset)
print(DIST_raster)
#%%
## CHECKPOINT
DIST_raster = "projects/pyregence-ee/assets/op-tx/test3/outputs/Treatments_Plumas_Protect_Alt1_DIST"
#%%
## CHECKPOINT
# poll for tasks
cg_ic = cg(DIST_raster,poll=True)
print(cg_ic)
#%%
# don't poll for tasks, but check that cg_ic.size() == fm40_ic.size(), return when true
fm40_ic = fm40(DIST_raster,fuels_source='pyrologix')
print(fm40_ic)
#%%
# poll for tasks
cc,ch = cc_ch(DIST_raster,fuels_source='pyrologix',poll=True)
print(cc,ch)
#%%
# CHECKPOINT
cc = "projects/pyregence-ee/assets/op-tx/test3/outputs/Treatments_Plumas_Protect_Alt1_fuelscape/CC"
ch = "projects/pyregence-ee/assets/op-tx/test3/outputs/Treatments_Plumas_Protect_Alt1_fuelscape/CH"
#%%
# poll for tasks
cbd,cbh = cbd_cbh(DIST_raster,fuels_source='pyrologix',poll=True)
print(cbd,cbh)
#%%
# poll for task
#CHECKPOINT
gcs_fuelstack = export_fuelstack(folder="projects/pyregence-ee/assets/op-tx/test3/outputs/Treatments_Plumas_Protect_Alt1_fuelscape",poll=True)
# %%
# poll for tasks
# CHECKPOINT
local_fuelstack = download_gcs(gcs_fuelstack)