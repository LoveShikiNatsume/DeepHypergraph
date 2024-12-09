from functools import partial
from typing import Optional

from dhg.data import BaseData
from dhg.datapipe import load_from_pickle, to_bool_tensor, to_long_tensor, to_tensor, norm_ft

class MicroServiceHypergraph(BaseData):
    r"""The MicroServiceHypergraph dataset for vertex classification task.
    It is a hypergraph dataset, in which vertex denotes the microservice instance and hyperedge denotes
    the deployment relationship between microservices. Each microservice is also associated with category information.

    The content of the MicroServiceHypergraph dataset includes the following:

    - ``num_classes``: The number of classes (not specified in the original dataset, set to 1 for simplicity).
    - ``num_vertices``: The number of vertices: :math:`10`.
    - ``num_edges``: The number of edges: :math:`3`.
    - ``edge_list``: The edge list. ``List`` with length :math:`3`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(10)`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(10)`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(10)`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(10)`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("micro_service_hypergraph", data_root)
        self._content = {
            "num_classes": 5,  # Assuming a single class for simplicity
            "num_vertices": 10,
            "num_edges": 4,
            "edge_list": {
                "upon": [
                    {
                        "filename": "edge_list.pkl",
                        "md5": "2fd63daecbf454847a9f75b3aa28d0e2",  # Replace with actual MD5 checksum
                    }
                ],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [
                    {
                        "filename": "labels.pkl",
                        "md5": "ee872eade9fc4dfd96cd31522efe36e7",  # Replace with actual MD5 checksum
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "features": {
                "upon": [
                    {
                        "filename": "features.pkl",
                        "md5": "52f2fdc8cea8c1c6c51bd667cac0d626",  # Replace with actual MD5 checksum
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "train_mask": {
                "upon": [
                    {
                        "filename": "train_mask.pkl",
                        "md5": "922a536e6253f11adf0eb24497fb849a",  # Replace with actual MD5 checksum
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [
                    {
                        "filename": "val_mask.pkl",
                        "md5": "79e539f525bfcee202b6e40d870654ad",  # Replace with actual MD5 checksum
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [
                    {
                        "filename": "test_mask.pkl",
                        "md5": "8787515553f9052446cf4e8b09c3ad64",  # Replace with actual MD5 checksum
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }