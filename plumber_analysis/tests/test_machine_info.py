import unittest
import tempfile

from plumber_analysis import machine_info

class TestMachineInfo(unittest.TestCase):
    def test_machine_info(self):
        machine_dict = {
            "HOSTNAME": "localhost",
            "NUM_CORES": 16,
            "MEMORY": 16,
            "FILES": None,
        }
        list_of_dict = [machine_dict]
        info = machine_info.MachineInfo.from_list_of_dict(list_of_dict)
        self.assertEqual(info.machines(), list_of_dict)

if __name__ == "__main__":
    unittest.main()
