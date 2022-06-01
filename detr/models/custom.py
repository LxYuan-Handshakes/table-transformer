"""
Custom model script to load TableRecognizer, which composed of two models:
TableDetection and TableStructure models, for demo purpose.
"""
import torch
import torchvision.transforms.functional as F
from torch import nn
from PIL import Image
import pandas as pd
import cv2

from util import helpers
from datasets.transforms import Normalize
from typing import Union, Tuple, Type

BoundingBox = Tuple[int, int, int, int]


class TableDetector(nn.Module):
    """Initialise table detection model using a predefined config file and
    pretrained weight file.

    Usage example:

        Case 1: To load custom config and model weight

        >> table_detection_model = TableDetector(my_config_path, my_model_weight_path)
        >> table_bounding_box = table_detection_model.predict(image_path)
        [(150, 1033, 1200, 1239)]


        Case 2: To load official Microsoft config and pretrained model weight

        >> table_detection_model = TableDetector()
        >> table_bounding_box = table_detection_model.predict(image_path)
        [(150, 1033, 1200, 1239)]

        The returned table bounding box can then be used by the TableStructure
        model to crop the table regions out and to recognise table structure
        (i.e., rows and column).

    """

    def __init__(
        self,
        path_to_config: Union[str, None] = None,
        path_to_weight: Union[str, None] = None,
    ):
        super().__init__()

        if path_to_config is None:
            path_to_config = "./src/detection_config.json"

        if path_to_weight is None:
            path_to_weight = "./pretrained_models/pubtables1m_detection_detr_r18.pth"

        self.config_path = path_to_config
        self.weight_path = path_to_weight

        args = helpers.load_args(path_to_config)
        args["model_load_path"] = self.weight_path
        args = type("Args", (object,), args)

        self.device = "cpu"
        self.model, _, self.postprocessors = helpers.get_model(args, self.device)
        self.model.eval()

        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.class_mapping = {"table": 0, "table rotated": 1, "no object": 2}

    def get_class_mapping(self, key: str = "name"):
        """Return TableDetection class mapping. Set key as `name` or `key` to
        get label2idx or idx2label like mapping, respectively."""

        assert key in ["index", "name"]

        if key == "name":
            return {"table": 0, "table rotated": 1, "no object": 2}
        else:
            return {0: "table", 1: "table rotated", 2: "no object"}

    def predict(
        self, image_path: str, threshold: float = 0.5, padding: int = 50
    ) -> list[BoundingBox]:
        """Inference method for TableDetection model and to return list of
        table bounding-box if predicted score is equal or greather than
        threshold, empty list otherwise."""

        image = image_path
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
            image = helpers.add_padding(image, padding=padding)

        w, h = image.size

        img_tensor = self.normalize(F.to_tensor(image))[0]
        img_tensor = torch.unsqueeze(img_tensor, 0).to(self.device)

        outputs = None
        with torch.no_grad():
            outputs = self.model(img_tensor)

        image_size = torch.unsqueeze(torch.as_tensor([int(h), int(w)]), 0).to(
            self.device
        )
        results = self.postprocessors["bbox"](outputs, image_size)[0]

        bounding_box_table = [
            tuple(map(int, boxes))
            for score, label, boxes in zip(
                results["scores"], results["labels"], results["boxes"]
            )
            if score >= threshold and label == self.get_class_mapping()["table"]
        ]

        return bounding_box_table


class TableStructure(nn.Module):
    """Initialise table structure model using a predefined config file and
    pretrained weight file.

    """

    def __init__(
        self,
        path_to_config: Union[str, None] = None,
        path_to_weight: Union[str, None] = None,
    ):
        super().__init__()

        if path_to_config is None:
            path_to_config = "./src/structure_config.json"

        if path_to_weight is None:
            path_to_weight = "./pretrained_models/pubtables1m_structure_detr_r18.pth"

        self.config_path = path_to_config
        self.weight_path = path_to_weight

        args = helpers.load_args(path_to_config)
        args["model_load_path"] = self.weight_path
        args = type("Args", (object,), args)

        self.device = "cpu"
        self.model, _, self.postprocessors = helpers.get_model(args, self.device)
        self.model.eval()

        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def get_class_mapping(self, key: str = "name"):
        """Return TableStructure class mapping. Set key as `name` or `key` to
        get label2idx or idx2label like mapping, respectively."""

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

    def predict(
        self,
        image_path: str,
        table_region: BoundingBox,
        threshold: float = 0.7,
        padding: int = 50,
        debug: bool = True,
    ) -> pd.DataFrame:
        """Inference method for TableStructure model and to a dataframe."""

        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
            xmin, ymin, xmax, ymax = table_region
            crop_region = (
                xmin - padding,
                ymin - padding,
                xmax + padding,
                ymax + padding,
            )
            image = image.crop(crop_region)
            image = helpers.add_padding(image, padding=padding)

        w, h = image.size

        img_tensor = self.normalize(F.to_tensor(image))[0]
        img_tensor = torch.unsqueeze(img_tensor, 0).to(self.device)

        #   inference
        outputs = None
        with torch.no_grad():
            outputs = self.model(img_tensor)

        image_size = torch.unsqueeze(torch.as_tensor([int(h), int(w)]), 0).to(
            self.device
        )
        results = self.postprocessors["bbox"](outputs, image_size)[0]

        #   conversion to objects w/ score threshold
        objs = helpers.predictions_to_objects(
            results, threshold, self.get_class_mapping(key="index")
        )

        #   align columns and rows to table border
        xmin, ymin, xman, ymax = padding, padding, w - padding, h - padding
        table_bbox = [xmin, ymin, xman, ymax]
        objs = helpers.border_align(objs, table_bbox)

        #   fix overlapping and align objects
        objs = helpers.structure_table(objs, table_bbox)

        #   keep only the columns and rows
        objs = helpers.filter_cols_and_rows(objs)

        #   get cells based on cols and rows
        cells = helpers.get_cells(objs)
        helpers.set_cell_text(cells, image, clean=True)
        df = helpers.cells_to_dataframe(cells)

        if debug:
            visualization = helpers.visualize_structure(image, objs)
            out_path = image_path[:-4] + "_visualisation.jpg"
            cv2.imwrite(out_path, visualization)
            print(f"Visualization can be found at '{out_path}'.")

        return df


class TableRecogniser(nn.Module):
    """Initialise TableRecogniser model to detect and recognize table structure
    and finally return list of dataframe.

    Usage example:

        >> model = TableRecognizer()
        >> output = model.predict(image_path)
        >> len(output)
        1

        >> output[0].head(30)

    """

    def __init__(
        self,
        detector_model: Union[Type[TableDetector], None] = None,
        structure_model: Union[Type[TableStructure], None] = None,
    ):
        super().__init__()

        if detector_model is None:
            detector_model = TableDetector()

        if structure_model is None:
            structure_model = TableStructure()

        self.detector_model = detector_model
        self.structure_model = structure_model

    def predict(
        self,
        image_path: str,
        detector_threshold: float = 0.5,
        structure_threshold: float = 0.7,
    ) -> list[pd.DataFrame]:
        """Inference method to call TableDetector.predict() and pass the
        output to TableStructure.predict() and return list of dataframe."""

        bounding_box_table = self.detector_model.predict(image_path, detector_threshold)

        outputs = []
        for table_bbox in bounding_box_table:
            dataframe = self.structure_model.predict(
                image_path, table_bbox, structure_threshold
            )
            outputs.append(dataframe)

        return outputs
