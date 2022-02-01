"""Machine info for optimization tunables."""

import yaml
import multiprocessing as mp

class MachineInfo(object):
    """Class that tracks machine information for many machines"""
    REQUIRED_FIELDS = set(["HOSTNAME",
                           "NUM_CORES",
                           "MEMORY",
                           "FILES"])
    def __init__(self, list_of_dict):
        if not MachineInfo.validate_data(list_of_dict):
            raise ValueError("Failed to parse list_of_dict")
        # Each element in the list is a machine. Each element is a dict
        self.list_of_dict = list_of_dict

    def machines(self) -> list:
        """Use this to read and write machines."""
        return self.list_of_dict

    @staticmethod
    def read_configuration(self, filename: str):
        with open(filename) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return MachineInfo.from_list_of_dict(data)

    def write_configuration(self, filename: str):
        data = self.to_list_of_dict()
        with open(filename, "w") as f:
            yaml.dump(f, data)

    def to_list_of_dict(self) -> list:
        return self.list_of_dict

    @staticmethod
    def from_list_of_dict(list_of_dict: list) -> None:
        return MachineInfo(list_of_dict)

    @staticmethod
    def validate_data(list_of_dict: list) -> bool:
        for d in list_of_dict:
            if not MachineInfo.REQUIRED_FIELDS.issubset(d.keys()):
                return False
        return True


def generate_localhost_machine_dict() -> dict:
    """Use as a template to create machine_info"""
    num_cores = mp.cpu_count()
    from psutil import virtual_memory

    mem = virtual_memory()
    total_memory = mem.total  # total physical memory available
    machine_dict = {
        "HOSTNAME": "localhost",
        "NUM_CORES": num_cores,
        "MEMORY": total_memory,
        "FILES": None,
    }
    return machine_dict


class MachineClass(object):
    """For cloud instances, we may have a potentially (or practically)
    infinite set of
    configurations. For example, CPU and memory may be priced per byte or core
    seconds, and disks can be added in combinations.
    This class provides an interface to query these limits.
    """
    # TODO(mkuchnik): Provide interface for files
    def price_per_vCPU_hour() -> float:
        """Price in dollars per vCPU hour"""
        raise NotImplemented()
    def price_per_GB_hour() -> float:
        """Price in dollars per GB hour"""
        raise NotImplemented()


class GCPN1OnDemand(MachineClass):
    """Taken from N1 us-east1"""
    @staticmethod
    def price_per_vCPU_hour() -> float:
        """Price in dollars per vCPU hour"""
        return 0.031611
    @staticmethod
    def price_per_GB_hour() -> float:
        """Price in dollars per GB hour"""
        return 0.004237

class GCPLocalSSD(object):
    @staticmethod
    def price_per_GB_month() -> float:
        # https://cloud.google.com/compute/disks-image-pricing
        return 0.080

    @staticmethod
    def price_per_GB_hour() -> float:
        return GCPLocalSSD.price_per_GB_month() / 730

    @staticmethod
    def MBps_per_GB() -> float:
        """ Read bw vs storage space ratio """
        # https://cloud.google.com/compute/docs/disks/local-ssd#performance
        return 660 / 375

    @staticmethod
    def price_per_MBps_hour() -> float:
        return GCPLocalSSD.price_per_GB_hour() * GCPLocalSSD.MBps_per_GB()
