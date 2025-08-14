import glob
import os

input_path = os.path.join("결과", "v3")
output_path = "pred"
os.makedirs(output_path, exist_ok=True)

image_paths1 = [os.path.basename(x).split(".")[0] for x in glob.glob(os.path.join("dataset", "domain", "**", "*.jpg"))]
image_paths2 = [os.path.basename(x).split(".")[0] for x in glob.glob(os.path.join("dataset", "sha", "**", "*.jpg"))]
image_paths3 = [os.path.basename(x).split(".")[0] for x in glob.glob(os.path.join("dataset", "shb", "**", "*.jpg"))]
image_paths4 = [os.path.basename(x).split(".")[0] for x in glob.glob(os.path.join("dataset", "qnrf", "**", "*.jpg"))]
image_paths5 = [os.path.basename(x).split(".")[0] for x in glob.glob(os.path.join("dataset", "nwpu", "**", "*.jpg"))]
image_paths6 = [os.path.basename(x).split(".")[0] for x in glob.glob(os.path.join("dataset", "head2crowd", "**", "*.jpg"))]

prediction_by_dataset = dict()
with open(os.path.join(input_path, "prediction.txt"), "r") as txt_file:
    predictions = txt_file.read().splitlines()
    for prediction in predictions:
        image_name, gt, pred = prediction.split(" ")
        if image_name in image_paths1:
            dataset_name = "domain"
        elif image_name in image_paths2:
            dataset_name = "sha"
        elif image_name in image_paths3:
            dataset_name = "shb"
        elif image_name in image_paths4:
            dataset_name = "qnrf"
        elif image_name in image_paths5:
            dataset_name = "nwpu"
        elif image_name in image_paths6:
            dataset_name = "head"
        else:
            raise ValueError

        if dataset_name not in prediction_by_dataset.keys():
            prediction_by_dataset[dataset_name] = []
        prediction_by_dataset[dataset_name].append([image_name, gt, pred])

datasets = list(prediction_by_dataset.keys())
for dataset in datasets:
    with open(f"{output_path}/{dataset}.txt", "w") as txt_file:
        for image_name, gt, pred in prediction_by_dataset[dataset]:
            image_name = image_name.replace("uptec_head", "domain_normal")
            txt_file.write(f"{image_name} {gt} {int(float(pred))}\n")



