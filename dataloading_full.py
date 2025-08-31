# This script converted the test sets of both QG and TL models to root 
# to be used later for inference studies

from hepconvert import parquet_to_root
import os

parquet_to_root("./datasets/qg_dataset/QuarkGluon/qg_test_file_0.root", "./datasets/qg_dataset/QuarkGluon/test_file_0.parquet", name="tree", force=True)
parquet_to_root("./datasets/qg_dataset/QuarkGluon/qg_test_file_1.root", "./datasets/qg_dataset/QuarkGluon/test_file_1.parquet", name="tree", force=True)

parquet_to_root("./datasets/tl_dataset/TopLandscape/tl_test_file.root", "./datasets/tl_dataset/TopLandscape/test_file.parquet", name="tree", force=True)