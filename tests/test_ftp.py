import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import os

from src.utils.ftp_utils import upload_files_to_ftp, _ensure_remote_dir, test_ftp_connection


class TestFTPUtils(unittest.TestCase):

    def test_ensure_remote_dir(self):
        mock_ftp = MagicMock()
        _ensure_remote_dir(mock_ftp, "/models/checkpoints")
        mock_ftp.cwd.assert_any_call("/")
        mock_ftp.cwd.assert_any_call("models")
        mock_ftp.cwd.assert_any_call("checkpoints")

    @patch("src.utils.ftp_utils._get_ftp_connection")
    def test_upload_files_to_ftp_success(self, mock_get_conn):
        mock_ftp = MagicMock()
        mock_get_conn.return_value = (mock_ftp, "ftp.example.com")

        with tempfile.TemporaryDirectory() as tmp_dir:
            f1 = Path(tmp_dir) / "best.pth"
            f1.write_bytes(b"dummy checkpoint data")
            f2 = Path(tmp_dir) / "metrics.json"
            f2.write_text('{"mae": 1.25}')

            result = upload_files_to_ftp(
                files=[f1, f2],
                ftp_host="ftp.example.com",
                ftp_port=21,
                ftp_user="user",
                ftp_password="pass",
                remote_dir="/artifacts",
            )
            self.assertTrue(result)
            self.assertEqual(mock_ftp.storbinary.call_count, 2)
            mock_ftp.quit.assert_called_once()

    def test_upload_files_no_host(self):
        with patch.dict(os.environ, {"FTP_HOST": ""}, clear=False):
            result = upload_files_to_ftp(files=["/non/existent/file"], ftp_host="")
            self.assertFalse(result)

    @patch("src.utils.ftp_utils._get_ftp_connection")
    def test_test_ftp_connection(self, mock_get_conn):
        mock_ftp = MagicMock()
        mock_ftp.getwelcome.return_value = "220 Service ready"
        mock_ftp.pwd.return_value = "/"
        mock_get_conn.return_value = (mock_ftp, "ftp.example.com")

        result = test_ftp_connection(ftp_host="ftp.example.com")
        self.assertTrue(result)
        mock_ftp.quit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
