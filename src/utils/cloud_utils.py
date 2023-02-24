#%%
import os
import subprocess
from fnmatch import fnmatch
import zipfile
from pathlib import Path
import datetime
import logging
import time
import ee

try:
    credentials = ee.ServiceAccountCredentials(email=None,key_file='/home/private-key.json')
    ee.Initialize(credentials)
except:
    ee.Initialize(project="pyregence-ee")

repo_dir =  os.path.abspath(os.path.join(__file__ ,"../../..")) # three parents up
date_id = datetime.datetime.utcnow().strftime("%Y-%m-%d").replace('-','') # like 20221216
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.WARNING,
    filename = os.path.join(repo_dir,"log",f"{date_id}.log")
)
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

def zip_shp(shpfile:str):
    """
    zip .shp and associated files at current directory
    args: 
        shpfile (str): file path to shapefile
    return:
        path to zipped file (str)
    """
    # path = Path(shpfile).resolve()
    parent = os.path.dirname(shpfile)
    basename = os.path.basename(shpfile).split('.shp')[0]
    list_files  = [os.path.join(parent,i) for i in os.listdir(parent) if fnmatch(i,f'{basename}*')]
    logger.info(f"Adding files to .zip archive: {list_files}")
    zipfile_path = shpfile.replace('.shp','.zip')
    with zipfile.ZipFile(zipfile_path, 'w') as zipF:
        for file in list_files:
            zipF.write(file,os.path.basename(file),compress_type=zipfile.ZIP_DEFLATED)
    return zipfile_path

def make_gee_folders(parent:str, children:list):
    """
    make necessary folders in GEE 
    
    NOTE: parent starts at projects/pyregence-ee/assets/op-tx/{parent}
    
    args:
        parent (str): parent folder name

        children (list): list of child folder names to create in parent
    returns:
        None
    """
    # GEE asset folder(s)
    # default base path is for op-tx
    optx_folder = "projects/pyregence-ee/assets/op-tx"
    gee_parent_folder = f"{optx_folder}/{parent}" 
    # children under project
    gee_project_children = [f"{gee_parent_folder}/{child}" for child in children]
    
    # make tree
    for f in [gee_parent_folder,
                *gee_project_children]:
        
        create_folder_cmd = f"earthengine create folder {f}"
        logger.info(create_folder_cmd)
        proc = subprocess.Popen(
                create_folder_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
        out, err = proc.communicate()
    return

def upload_gcs(local_file:str,gcs_project_folder:str,):
    """
    Upload files to new or existing location on a GCP Bucket
    args:
        local_file (str): local path to treatments shapefile
        
        gcs_project_folder (str): parent folder on GCS to upload local file to
    returns:
        destination file path on Google Cloud Storage bucket (str)
    """
    gcs_dest = f"{gcs_project_folder}/{os.path.basename(local_file)}"
    gsutil_upload_cmd = f"gsutil cp {local_file} {gcs_dest}"
    logger.info(gsutil_upload_cmd)
    proc = subprocess.Popen(
            gsutil_upload_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    out, err = proc.communicate()
    return gcs_dest

def upload_gee(gcs_file:str,gee_project_folder:str):
    """
    Upload files from Google Cloud Storage to GEE assets
    args:
        gcs_file (str): local path to treatments shapefile
        
        gee_project_folder (str): GEE asset folder to upload asset to 
    returns:
        the file's asset path in GEE (str)
    """    
    
    #upload GCS file to GEE in the asset folder
    gee_asset_path = f"{gee_project_folder}/{os.path.basename(gcs_file).split('.')[0]}"
    gee_upload_cmd = f"earthengine upload table --asset_id={gee_asset_path} {gcs_file}"
    logger.info(gee_upload_cmd)
    proc = subprocess.Popen(
            gee_upload_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    out, err = proc.communicate()
    return gee_asset_path

def download_gcs(gcs_file):
    """
    pulls files from google cloud storage to local machine
    args:
        gcs_file (str): full qualified GCS path to file starting with 'gs://'
    returns:
        local path to file (str)
    """
    outputs_dir = Path(os.path.join(repo_dir, 'data','outputs')).resolve()
    if not outputs_dir.exists():
        outputs_dir.mkdir(parents=True)
    
    local_file = os.path.join(outputs_dir,os.path.basename(gcs_file))
    gsutil_cmd = f"gsutil cp {gcs_file} {local_file}"
    logger.info(gsutil_cmd)
    proc = subprocess.Popen(
            gsutil_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    out, err = proc.communicate()
    return local_file

# this polling function handles batch export tasks once they ahve been started 
# allows precise polling of individual tasks within scope of their generation
def poll_submitted_task(task,sleeper:int):
    """
    polls for the status of one started task, completes when task status is 'COMPLETED'
    args:
        task : task status dictionary returned by ee.batch.Task.status() method
        sleeper (int): minutes to sleep between status checks
    returns:
        None
    """
    # handles instances of ee.batch.Task, 
    # NOTE: task needs to be started in order to retrieve needed status info, 
    # so use this function after doing `task.start()`
    if isinstance(task,ee.batch.Task):
        t_id = task.status()['id']
        status = task.status()['state']
        
        if status == 'UNSUBMITTED':
            raise RuntimeError(f"run .start() method on task before polling. {t_id}:{status}")
        
        logger.info(f"polling for task: {t_id}")
        while status != 'COMPLETED':
            if status in ['READY','RUNNING']:
                logger.info(f"{t_id}:{status} [sleeping {sleeper} mins] ")
                time.sleep(60*sleeper)
                t_id = task.status()['id']
                status = task.status()['state'] 
            elif status in ['FAILED','CANCELLED','CANCEL_REQUESTED']:
                raise RuntimeError(f"problematic task status code - {t_id}:{status}")
        logger.info(f"{t_id}:{status}")
    else:
        raise TypeError(f"{task} is not instance of <ee.batch.Task>. {type(task)}")
    return

# these two functions handle tasks as generated from 
# earthengine CLI (e.g. earthengine task list)
# good for polling for tasks outside scope of their generation
def ee_task_list_complete(desc:str,items:int):
    """
    Tests if particular type of task(s) have COMPLETED on EE server
    args:
        desc (str): description of task type in 2nd column of earthengine task list output
        items (int): number of task items to check on (descending by date submitted)
    returns:
        list (e.g. [True,True,False])
    """
    # parse earthengine task list output into python list
    ee_task_list = os.popen(f"earthengine task list").read().split('\n')
    upload_tasks = [t for t in ee_task_list if desc in t][0:items]
    # test each item's status reads COMPLETE
    test_complete = ['COMPLETED' in t for t in upload_tasks]
    return test_complete

def ee_task_list_poller(desc:str,items:str,sleep:int):
    """
    Polls for tasks running on EE server, fetched by earthengine task list CLI command
    args:
        desc (str): one of Upload,Export.image, others?..
        items (int): number of upload tasks to poll for
        sleep (int): number of minutes to sleep in between checks
    returns:
        None
    """
    #parses earthengine task list output
    test_complete = ee_task_list_complete(desc,items)
    while not all(test_complete):
        logger.info(f"Waiting for one or more {desc} tasks to complete: {test_complete}. Sleeping {sleep} mins..")
        time.sleep(60*sleep)
        # Could fetch Upload, Export.image, other types of tasks 
        # as presented in 2nd column of earthengine task list output
        test_complete = ee_task_list_complete(desc,items)
    
    logger.info(f'all {desc} complete')
    return
