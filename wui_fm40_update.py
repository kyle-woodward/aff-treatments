"""Builds Random Forest Model based on previously export training data
    then updates the FM40 with the model inference
"""
from sklearn import metrics
import pandas as pd
import ee
import os
import yaml
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.WARNING,
    filename=os.path.join(os.path.dirname(__file__),'wui_fm40_update.log')
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ee.Initialize()

def main():
    """Main level function for updating FireFactor FM40 in WUI based on WUI RF"""
    
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

    args = parser.parse_args()

    # parse config file
    with open(args.config) as file:
        config = yaml.full_load(file)

    geo_info = config["geo"]
    version = config["version"].get('latest')

    # extract out geo information from config
    geo_t = geo_info["crsTransform"]
    scale = geo_info["scale"]
    dims = geo_info["dimensions"]
    crs = geo_info["crs"]

    # Create a simple LS composite for each year in a list
    def preprocess_landsat(image):
        qaband = image.select('QA_PIXEL')
        cirrusMask = qaband.bitwiseAnd(1 << 2).eq(0)
        cloudMask = qaband.bitwiseAnd(1 << 3).eq(0)
        shadowMask = qaband.bitwiseAnd(1 << 4).eq(0)
        snowMask = qaband.bitwiseAnd(1 << 5).eq(0)
        saturationMask = image.select('QA_RADSAT').eq(0);
        qaMask = cloudMask.And(cirrusMask).And(shadowMask).And(snowMask)
        mask = qaMask.And(saturationMask)
        img_bands = (
            image
            .select(
                ['SR_B.*'],
                ['coastal',"blue", "green", "red", "nir", "swir1", "swir2"]
            )
            .multiply(0.0000275)
            .add(-0.2)
        )
        ndvi = img_bands.normalizedDifference(["nir","red"]).rename("ndvi")
        mndwi = img_bands.normalizedDifference(["green","swir1"]).rename("mndwi")
        bai = img_bands.expression("1/((0.1-b('red'))**2 +(0.06-b('nir'))**2)").rename("bai")
        return ee.Image.cat([img_bands,ndvi,mndwi,bai]).updateMask(mask)


    # import fm40 and define classes
    fm40 = ee.ImageCollection('projects/pyregence-ee/assets/conus/landfire/fbfm40')
    fm40_2019 = fm40.filter(ee.Filter.eq("system:index", "LF2019_FBFM40_200")).first()
    fm40_classes = [
        0,
        90,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        121,
        122,
        123,
        124,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        161,
        162,
        163,
        164,
        165,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        201,
        202,
        203,
        204,
    ]

    fm40_cls_pts = [250 for i in fm40_classes]

    # import WUI layer and define sample zones
    wui_zones = ee.Image("projects/pyregence-ee/assets/conus/vulnerability/wui_v3") # 1.0 mi buffer
    # to get region export
    cc_ic = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/cc")
    cc_img = ee.Image(
            cc_ic.filter(ee.Filter.eq("version", 200)).limit(1, "system:time_start").first()
        )
    # Import Layers for Feature Stack
    mtbs_image_collection = ee.ImageCollection('projects/pyregence-ee/assets/conus/mtbs/burn-severity-corrected-11172021') #sub in stripe-corrected burn severity collection
    total_dest_img = ee.Image("projects/pyregence-ee/assets/conus/vulnerability/wui-training-fires/sit209-total_dest_buffered")

    # decided not to use these as they are predictively unimportant (drag down accuracy) and limit sample area 
    # yr_blt = ee.ImageCollection("projects/pyregence-ee/assets/conus/vulnerability/yr_blt_v1").mosaic().unmask(0).rename("yr_blt")
    # assd_val = ee.ImageCollection("projects/pyregence-ee/assets/conus/vulnerability/assd_val_v1").mosaic().unmask(0).rename("assd_val")
    # luse_code = ee.ImageCollection("projects/pyregence-ee/assets/conus/vulnerability/usecode_n_v1").mosaic().unmask(-1).rename("luse_code")
    # sq_ft = ee.ImageCollection("projects/pyregence-ee/assets/conus/vulnerability/sqft_v1").mosaic().unmask(0).rename("sq_ft")
    # roof_type = ee.ImageCollection("projects/pyregence-ee/assets/conus/vulnerability/rooftype_n_v1").mosaic().unmask(-1).rename("roof_type")
    # ext_wall = ee.ImageCollection("projects/pyregence-ee/assets/conus/vulnerability/ext_wall_n_v1").mosaic().unmask(-1).rename("ext_wall")
    # def_space = ee.ImageCollection("projects/pyregence-ee/assets/conus/vulnerability/dspace_v1").mosaic().unmask(0).rename("def_space")
    # dspace_nn = ee.ImageCollection("projects/pyregence-ee/assets/conus/vulnerability/nn_dspace_v1").mosaic().unmask(0).rename("dspace_nn")

    bldg_density = ee.ImageCollection("projects/pyregence-ee/assets/conus/vulnerability/bldg_density_km2_v1").mosaic().unmask(0).rename("bldg_density")
    severity = ee.Image("projects/pyregence-ee/assets/conus/vulnerability/wui-training-fires/burn_severity_buffered").rename("SEVERITY")
    dem = ee.Image("USGS/3DEP/10m")
    topo = ee.Algorithms.Terrain(dem).select(["elevation","slope","aspect"])
    fc = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fvc").filter(ee.Filter.eq("version",200)).first()
    fh = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fvh").filter(ee.Filter.eq("version",200)).first()
    vt = ee.ImageCollection("projects/pyregence-ee/assets/conus/landfire/fvt").filter(ee.Filter.eq("version",200)).first()
    ls = ee.Image("projects/pyregence-ee/assets/conus/vulnerability/wui-training-fires/landsat_mosaic_idx_bai")
    clim = ee.Image("projects/pyregence-ee/assets/conus/vulnerability/wui-training-fires/gridmet-all_idx_bi")


    # Construct Training/Testing samples from previously exported fm40 sample points
    samples = ee.FeatureCollection("projects/pyregence-ee/assets/conus/vulnerability/wui-training-fires/fm40_sample_points_features_wui_v3")
    samples = samples.filter(ee.Filter.neq("FBFM40",90))

    training = samples.filter(ee.Filter.lte("random",0.7))
    testing = samples.filter(ee.Filter.gt("random",0.7))

    # Create training stack
    training_bands = ee.Image.cat([
        ls,
        clim,
        total_dest_img,
        topo,
        bldg_density,
        fc,
        fh,
        vt,
        severity,
    ])

    def kfold_ee_classifier(i):
        """Random Forest Classifier K-fold training and testing"""
        def stratified_random_sample(j):
            j = ee.Number(j)
            class_tbl = training.filter(ee.Filter.eq("FBFM40",j))
            return class_tbl.randomColumn("random_strata", seed=i.add(1).multiply(2))
        
        i = ee.Number(i)
        
        features = training_bands.bandNames()
        label = 'FBFM40'
        
        shuffled = ee.FeatureCollection(ee.List(fm40_classes).map(stratified_random_sample)).flatten()
        
        fold_training = shuffled.filter(ee.Filter.lte("random_strata",0.8))
        fold_testing = shuffled.filter(ee.Filter.gt("random_strata",0.8))

        #Train a random forest classifier
        classifier = (
            ee.Classifier.smileRandomForest(
                50, 
                minLeafPopulation=2, 
                maxNodes= 500, 
                bagFraction=0.7,
                seed=i.pow(2)
            )
            .train(
                features = fold_training, 
                classProperty = label, 
                inputProperties = features
            )
        )

        # Classify the image with the same bands used for training        
        test_classified = fold_testing.classify(classifier,)
        
        y_true = test_classified.aggregate_array("FBFM40").getInfo()
        #print(len(y_true))
        y_pred = test_classified.aggregate_array("classification").getInfo()
        #print(len(y_pred))
        
        acc = metrics.accuracy_score(y_true,y_pred)
        prec = metrics.precision_score(y_true,y_pred,average="weighted")
        reca = metrics.recall_score(y_true,y_pred,average="weighted")
        f1 = metrics.f1_score(y_true,y_pred,average="weighted")
        
        return acc, prec, reca, f1, fold_training, fold_testing, classifier
    
    logger.info('Training Model')
    
    # Train and Validate Classifier, craeting trained model and its metrics for each k-fold iteration
    accs = []
    precs = []
    recas = []
    f1s = []
    training_samples = []
    testing_samples = []
    models = []

    for i in range(10):
        acc, prec, reca, f1, fold_training, fold_testing, fold_model = kfold_ee_classifier(i)
        accs.append(acc)
        precs.append(prec)
        recas.append(reca)
        f1s.append(f1)
        training_samples.append(fold_training)
        testing_samples.append(fold_testing)
        models.append(fold_model)


    # make metrics dataframe
    kfold_df = pd.DataFrame({"accuracy":accs,"precision":precs,"recall":recas,"f1":f1s,"fold":range(1,11)})

    # select best of 10 fold models based on accuracy metric
    best_model_idx = kfold_df.sort_values(by="accuracy")["fold"].values[-1]-1
    best_model_idx

    best_model = models[best_model_idx]
    #print(best_model.getInfo())
    logger.info(f'Best Model Accuracy: {kfold_df.iloc[best_model_idx,0]}')

    # Create inference data 
    start_date = "2020-01-01"
    end_date = "2021-01-01"
    gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
    variables = ee.List(['tmmn', 'tmmx', 'vs', 'erc', 'bi', 'fm100', 'fm1000'])
    gridmet_vars = gridmet.select(variables) # subset gridmet bands to contents of "variable" list

    # gridmet_infer = gridmet_vars.filterDate(start_date,end_date).mean()
    gridmet_infer = gridmet_vars.filterDate(start_date,end_date).qualityMosaic("bi")
    conus = (
        ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        .filter(ee.Filter.eq("country_co", "US"))
        .geometry(1e4)
    )
    lc8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    )

    # lc8_infer = lc8.filterDate(start_date,end_date).filterBounds(conus).map(preprocess_landsat).median()
    lc8_infer = lc8.filterDate(start_date,end_date).filterBounds(conus).map(preprocess_landsat).qualityMosaic("bai")
    severity = mtbs_image_collection.filterDate("2019-01-01","2020-01-01").first().unmask(1)

    # Create inference stack
    infer_data = ee.Image.cat([
        lc8_infer,
        gridmet_infer,
        total_dest_img.unmask(0),
        topo,
        #yr_blt,
        #luse_code,
        #sq_ft,
        #roof_type,
        #ext_wall,
        #def_space,
        #dspace_nn,
        bldg_density,
        fc,
        fh,
        vt,
        severity
    ])
    
    
    inference_img = infer_data.classify(best_model).rename("FBFM40_infer")

    # import FireFactor FM40 (imgColl.mosaic() of calc_fm40.py output)
    fm40_FF = ee.Image(f"projects/pyregence-ee/assets/conus/fuels/Fuels_FM40_preWUIupdate_{version}").select('new_fbfm40')

    # Create replacement areas from WUI
    replace = wui_zones.gt(1).And(fm40_FF.eq(91).Or(fm40_FF.eq(93))) # FM40 91 and 93 that are in Intermix and interface zones

    # replace fm40 with inference image
    final = fm40_FF.where(replace, inference_img).rename("FBFM40_WUI").uint16()

    #export WUI-updated FM40
    task = ee.batch.Export.image.toAsset(
    image= final,
    description= f'FBFM40 WUI RF Update',
    assetId= f'projects/pyregence-ee/assets/conus/fuels/Fuels_FM40_{version}',
    dimensions= dims,
    region = cc_img.geometry(),
    crs= crs, 
    crs_transform= geo_t,
    maxPixels= 1e12,
    pyramidingPolicy = {".default":"mode"}
    )
    task.start()
    
    logger.info(f'Exporting wui-updated FM40 as ~/conus/fuels/Fuels_FM40_{version}')

if __name__ == "__main__":
    main()