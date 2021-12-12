import config
import unittest
from dataset import *
import os.path
class TestDataset(unittest.TestCase):
    def validate_dataset(self, dataset_df):
        self.assertEqual(dataset_df.columns.shape[0], 2)
        self.assertEqual(dataset_df['label'].nunique(), 4)
        self.assertEqual(dataset_df.shape[0], 422937)

    def test_download_dataset(self):

        path_zip_dataset =get_path_zip_dataset()
        url_dataset = get_url_dataset()
        download_dataset(url_dataset,path_zip_dataset)
        self.assertTrue(os.path.exists(path_zip_dataset))


    def test_get_dataset(self):
        get_dataset()
        path_source_dataset=get_path_source_dataset()
        self.assertTrue(os.path.exists(path_source_dataset))

    def test_unpack_dataset(self):
        path_zip_dataset =get_path_zip_dataset()
        path_dataset_dir  =get_path_dataset_dir()
        url_dataset = get_url_dataset()
        download_dataset(url_dataset,path_zip_dataset)
        unpack_dataset(path_zip_dataset,path_dataset_dir)
        self.assertTrue(os.path.exists(path_dataset_dir))
        remove_zip()

    def test_preprocess_dataset(self):

        preprocess_dataset()
        path_preprocessed_dataset = get_path_preprocessed_dataset()

        dataset_df=pd.read_csv(path_preprocessed_dataset,sep="\t",encoding="utf-8")
        self.assertTrue(os.path.isfile(path_preprocessed_dataset))
        self.validate_dataset(dataset_df)

    def test_load_dataset(self):
        preprocess_dataset()
        dataset_df = load_dataset()
        self.validate_dataset(dataset_df=dataset_df)
