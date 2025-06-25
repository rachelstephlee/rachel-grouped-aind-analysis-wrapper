import json
import pathlib
import unittest
from unittest.mock import MagicMock, patch

from analysis_wrapper.utils import read_input_json


class TestReadInputJson(unittest.TestCase):
    @patch("pathlib.Path.glob")  # Mock the glob method on Path
    @patch("builtins.open")  # Mock the open function
    @patch("json.load")  # Mock json.load function
    def test_read_input_json_success(self, mock_json_load, mock_open, mock_glob):
        # Prepare the mock objects
        mock_glob.return_value = [
            pathlib.Path("/some/directory/file.json")
        ]  # Simulate the path returned by glob
        mock_open.return_value.__enter__.return_value = (
            MagicMock()
        )  # Simulate the open context manager
        mock_json_load.return_value = {
            "location_bucket": "s3://s3://codeocean-s3datasetsbucket-1u41qdg42ur9/7f1eaf10-01bc-41cb-bc88-6464d0425b51",
            "location_asset_id": "7f1eaf10-01bc-41cb-bc88-6464d0425b51",
            "location_uri": "s3://codeocean-s3datasetsbucket-1u41qdg42ur9/7f1eaf10-01bc-41cb-bc88-6464d0425b51/nwb/behavior_769038_2025-02-10_13-16-09.nwb",
        }  # Simulate the JSON content

        # Call the function
        result = read_input_json()

        # Verify the result
        self.assertEqual(
            result,
            {
                "location_bucket": "s3://s3://codeocean-s3datasetsbucket-1u41qdg42ur9/7f1eaf10-01bc-41cb-bc88-6464d0425b51",
                "location_asset_id": "7f1eaf10-01bc-41cb-bc88-6464d0425b51",
                "location_uri": "s3://codeocean-s3datasetsbucket-1u41qdg42ur9/7f1eaf10-01bc-41cb-bc88-6464d0425b51/nwb/behavior_769038_2025-02-10_13-16-09.nwb",
            },
        )

        # Verify the interactions
        mock_glob.assert_called_once_with("*.json")
        mock_open.assert_called_once_with(
            pathlib.Path("/some/directory/file.json"), "r"
        )
        mock_json_load.assert_called_once()

    @patch("pathlib.Path.glob")
    def test_no_json_file_found(self, mock_glob):
        # Simulate no JSON files found
        mock_glob.return_value = []

        # Call the function and assert that the FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            read_input_json()

    @patch("pathlib.Path.glob")  # Mock the glob method on Path
    @patch("builtins.open")  # Mock the open function
    @patch("json.load")  # Mock json.load function
    def test_json_loading_error(self, mock_json_load, mock_open, mock_glob):
        # Simulate the presence of a JSON file
        mock_glob.return_value = [pathlib.Path("/some/directory/file.json")]
        mock_open.return_value.__enter__.return_value = MagicMock()

        # Simulate a JSON loading error (invalid JSON)
        mock_json_load.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        # Call the function and assert that the JSONDecodeError is raised
        with self.assertRaises(json.JSONDecodeError):
            read_input_json()


if __name__ == "__main__":
    unittest.main()
