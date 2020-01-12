#modified from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests
import zipfile
from pathlib import Path
import shutil


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                
def create_train_test_folder(destination):
    Path(destination + '\\NEU-DET\\train').mkdir(parents=True, exist_ok=True)
    Path(destination + '\\NEU-DET\\test').mkdir(parents=True, exist_ok=True)

def copy_img_and_notation(destination):
    start_train_index = 1
    end_train_index = 211
    start_test_index = 211
    end_test_index = 301
    annotation_source = destination + '\\NEU-DET\\ANNOTATIONS\\'
    img_source = destination + '\\NEU-DET\\IMAGES\\'
   
    defect_types = ['crazing_', 'inclusion_', 'patches_', 'pitted_surface_', 'rolled-in_scale_','scratches_']
    
    for idx in range(start_train_index, end_train_index):
        for defect in defect_types:
            shutil.copy2(annotation_source + defect + str(idx)+".xml", destination + 'NEU-DET\\train\\')
            shutil.copy2(img_source+ defect +str(idx)+".jpg", destination + 'NEU-DET\\train\\')
            
    for idx in range(start_test_index, end_test_index):
        for defect in defect_types:
            shutil.copy2(annotation_source + defect +str(idx)+".xml", destination + 'NEU-DET\\test\\')
            shutil.copy2(img_source + defect +str(idx)+".jpg", destination + 'NEU-DET\\test\\')
    

if __name__ == "__main__":
    file_id = '1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k'
    home = str(Path.home())
    destination = home + "\\.deeplearning4j\\data\\" #dont change this
    Path(destination).mkdir(parents=True, exist_ok=True)
    download_file_from_google_drive(file_id, destination+'NEU-DET.zip') #download zip files
    print("File saved to " + destination)
    with zipfile.ZipFile(destination+'NEU-DET.zip', 'r') as zip_ref:
        zip_ref.extractall(destination) #extract
    print("Extracting files to " + destination)
    create_train_test_folder(destination)
    print("copy files to training and test folders")
    copy_img_and_notation(destination)
