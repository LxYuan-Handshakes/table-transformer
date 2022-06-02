"""
Mostly modified from:
- https://github.com/phamquiluan/table-transformer/blob/main/core.py
- https://github.com/thibaultvt/table-transformer-simple-inference/blob/main/main.py

which made inference easier
"""

import torch
import json
import numpy as np
import cv2
import pandas as pd

from pytesseract import image_to_string

from PIL import Image

# sys.path.append("detr")
# sys.path.append("src")
from models import build_model


def load_args(json_path):
    """Loads arguments from JSON file

    Returns:
        data from the JSON file
    """
    data = None
    with open(json_path) as f:
        data = json.load(f)
    return data


def cells_to_dataframe(cells):
    d = {}
    for cell in cells:
        if cell["column"] not in d.keys():
            d[cell["column"]] = []
        d[cell["column"]].append(cell["content"])
    df = pd.DataFrame(data=d)
    return df


def set_cell_text(cells, image, clean=False):
    for cell in cells:
        #   crop & pad image
        xmin, ymin, xmax, ymax = cell["bbox"]
        roi = image.crop((xmin, ymin, xmax, ymax))
        roi = add_padding(roi, 30)

        #   OCR
        #   TODO: here you should use self.lang
        text = image_to_string(roi, lang="eng")
        if clean:
            text.strip()
            text = text.replace("\n", "")
        cell["content"] = text


def get_rows_and_columns(objs):
    cols = [obj for obj in objs if obj["label"] == "table column"]
    rows = [obj for obj in objs if obj["label"] == "table row"]
    #   sort cols bottom right x coordinate
    cols.sort(key=lambda col: col["bbox"][2])
    #   sort rows bottom right y coordinate
    rows.sort(key=lambda row: row["bbox"][3])

    return rows, cols


def get_cells(objs):
    rows, cols = get_rows_and_columns(objs)

    cells = []
    for i, col in enumerate(cols):
        c_xmin, c_ymin, c_xmax, c_ymax = col["bbox"]
        for j, row in enumerate(rows):
            r_xmin, r_ymin, r_xmax, r_ymax = row["bbox"]
            xmin, ymin = max(r_xmin, c_xmin), max(r_ymin, c_ymin)
            xmax, ymax = min(r_xmax, c_xmax), min(r_ymax, c_ymax)
            cell = {"column": i, "row": j, "bbox": [xmin, ymin, xmax, ymax]}
            cells.append(cell)

    return cells


#   Source file: postprocess.py
def structure_table(objs, table_bbox):
    rows, cols = get_rows_and_columns(objs)

    #   initial values are top and most left coordinates
    p_xmin, p_ymin, p_xmax, p_ymax = table_bbox
    p_ymax = p_ymin
    p_xmax = p_xmin
    """
       scenario 1: border is on same line []][]
           then: keep xmin
       scenario 2: border 1 is smaller than border 2, there is gap [] []
       scenario 3: border 1 is bigger than border 2, there is an overlap [[]]
           then: make xmin of current cell the xmax of the previous bbox
    """
    for row in rows:
        xmin, ymin, xmax, ymax = row["bbox"]
        if not p_ymax == ymin:
            ymin = p_ymax
        row["bbox"] = xmin, ymin, xmax, ymax
        p_ymax = ymax
    #   column bottom borders has to overlap with last row's bottom border
    # bottom_y = rows[-1]["bbox"][3]
    for col in cols:
        xmin, ymin, xmax, ymax = col["bbox"]

        if not p_xmax == xmin:
            xmin = p_xmax
        col["bbox"] = xmin, ymin, xmax, ymax
        p_xmax = xmax

    return objs


def filter_cols_and_rows(objs):
    """Filters anything else than columns and rows
    Returns:
        a list containing only column and row predictions
    """
    objs = [obj for obj in objs if obj["label"] in ["table column", "table row"]]
    return objs


def border_align(objs, table_bbox):
    """
    for every row and column,
    """
    for obj in objs:
        bbox = obj["bbox"]
        if obj["label"] == "table row":
            bbox[0] = table_bbox[0]
            bbox[2] = table_bbox[2]
        elif obj["label"] == "table column":
            bbox[1] = table_bbox[1]
            bbox[3] = table_bbox[3]
        obj["bbox"] = bbox

    return objs


def predictions_to_objects(predictions, threshold, class_map):
    objs = []
    labels = predictions["labels"].tolist()
    scores = enumerate(predictions["scores"].tolist())
    for idx, score in scores:
        if score > threshold:
            label = labels[idx]
            bbox = predictions["boxes"][idx].tolist()
            obj = {
                "score": score,
                "label": class_map[label],
                "bbox": list(map(int, bbox)),
            }
            objs.append(obj)
    return objs


def visualize_structure(image, objs):
    image = np.array(image)
    for obj in objs:
        xmin, ymin, xmax, ymax = obj["bbox"]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (245, 105, 66), 2)
    image = image[:, :, ::-1].copy()
    return image


#   Source file: postprocess.py (align_columns & align_rows)


def add_padding(img, padding=50):
    """Adds padding to an image
    Args:
        img (PIL.Image): pillow image
        padding (int): number of pixels to use for padding

    Returns:
        image with padding
    """

    w, h = img.size
    new_w = w + (padding * 2)
    new_h = h + (padding * 2)
    result = Image.new(img.mode, (new_w, new_h), (255, 255, 255))
    result.paste(img, (padding, padding))
    return result


#   Source file: main.py
def get_class_map(key="name"):
    assert key in ["index", "name"]
    if key == "name":
        return {
            "table": 0,
            "table column": 1,
            "table row": 2,
            "table column header": 3,
            "table projected row header": 4,
            "table spanning cell": 5,
            "no object": 6,
        }
    else:
        return {
            0: "table",
            1: "table column",
            2: "table row",
            3: "table column header",
            4: "table projected row header",
            5: "table spanning cell",
            6: "no object",
        }


# Source file: main.py
def get_model(args, device):
    """
    Loads DETR model on to the device specified.
    If a load path is specified, the state dict is updated accordingly.
    """
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.model_load_path:
        print(f"loading model from checkpoint: {args['model_load_path']}")
        loaded_state_dict = torch.load(args.model_load_path, map_location=device)
        model_state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in loaded_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=True)
    return model, criterion, postprocessors
