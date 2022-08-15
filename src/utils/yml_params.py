#%%
import yaml
import ee
ee.Initialize()

def get_export_params(yml):
    with open(yml) as file:
            config = yaml.full_load(file)

    geo_info = config["geo"]
    #print(geo_info)
    crs = geo_info["crs"]
    scale = geo_info["scale"]
    return crs,scale

# def exportImgToDrive(img,desc,folder,filenameprefix,):
#     task = ee.batch.Export.image.toDrive(image= img,
#                                             description= desc,
#                                             folder= folder,
#                                             fileNamePrefix= desc,
#                                             region = region,
#                                             scale = scale,
#                                             maxPixels= 1e13,
#                                             formatOptions={'cloudOptimized':True}
#                                             )

# %%
