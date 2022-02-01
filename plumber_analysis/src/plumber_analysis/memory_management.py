"""A tool to simulate high memory pressure.

The intent of this module is to eat memory to ensure the operating system has
little room to cache data and other shenanigans.
"""

import gc
import threading
import time

import psutil

class BloatingMemoryManager(object):
    def __init__(self, percentage_memory):
        """Constructs a memory manager which attempts to keep percentage_memory
        of the system full. For instance, if percentage_memory is set to 90%,
        and 50% of memory is used by other application, the memory manager will
        attempt to use 40% additional memory. This leaves 10% of memory before
        out of memory errors occur."""
        if percentage_memory < 0.0:
            raise ValueError("percentage_memory must be greater than 0.0.")
        elif percentage_memory > 1.0:
            raise ValueError("percentage_memory must be less than 1.0.")
        self.percentage_desired_used_memory = percentage_memory
        self.memory_container = []
        self.allocated_bytes = 0
        memory_stats = psutil.virtual_memory()
        self.total_memory = memory_stats.total
        self.force_gc = False
        self.print_fn = print

    def print(self, *args, **kwargs):
        # To overwrite
        return self.print_fn(*args, **kwargs)

    def percentage_available_system_memory(self):
        memory_stats = psutil.virtual_memory()
        return memory_stats.available / memory_stats.total

    def percentage_used_manager_memory(self):
        return self.allocated_bytes / self.total_memory

    def refresh_allocation(self):
        percentage_used_memory = 1. - self.percentage_available_system_memory()
        self.print("Percentage_used_memory: {}".format(percentage_used_memory))
        if percentage_used_memory < self.percentage_desired_used_memory:
            # Fill
            diff = self.percentage_desired_used_memory - percentage_used_memory
            allocate_bytes = int(self.total_memory * diff)
            self.allocate_num_bytes(allocate_bytes)
        else:
            # Free
            def get_free_bytes():
                percentage_used_memory = 1. - self.percentage_available_system_memory()
                diff = percentage_used_memory - self.percentage_desired_used_memory
                diff = max(diff, 0.)
                free_bytes = self.total_memory * diff
                return free_bytes
            self.print("Bytes wanted to free: {}".format(get_free_bytes()))
            self.print("Container size: {}".format(len(self.memory_container)))
            while (self.memory_container
                   and len(self.memory_container[-1]) < get_free_bytes()):
                freed_data = self.memory_container.pop(-1)
                self.allocated_bytes -= len(freed_data)
                del freed_data
            gc_limit = 4 * 10 ** 6
            if self.force_gc and (get_free_bytes() > gc_limit):
                self.print("Running gc")
                gc.collect()

        self.check_state_consistency()

    def check_state_consistency(self):
        assert self.allocated_bytes >= 0

    def allocate_num_bytes(self, num_bytes, chunk_size=None):
        """
        Allocates num_bytes using a chunk_size for allocations.
        chunk_size of 1MB is recommended.
        """
        if not isinstance(num_bytes, int):
            raise ValueError("Expected num_bytes to be an int, but got "
                             "{}".format(type(num_bytes)))
        if num_bytes == 0:
            return
        if chunk_size is None:
            chunk_size = int(10**6)
        num_chunks = num_bytes // chunk_size
        remainder_chunk = num_bytes % chunk_size
        mem_container = []
        for i in range(num_chunks):
            mem_container.append(' ' * chunk_size)
        if remainder_chunk:
            mem_container.append(' ' * remainder_chunk)
        self.memory_container.extend(mem_container)
        self.allocated_bytes += num_bytes

class AsyncBloatingMemoryManager(object):
    """Asynchronously runs the memory manager"""
    def __init__(self, percentage_memory, refresh_interval, delay=None):
        """Constructs a memory manager which attempts to keep percentage_memory
        of the system full. For instance, if percentage_memory is set to 90%,
        and 50% of memory is used by other application, the memory manager will
        attempt to use 40% additional memory. This leaves 10% of memory before
        out of memory errors occur."""
        self.memory_manager = BloatingMemoryManager(percentage_memory)
        self.refresh_interval = refresh_interval # in seconds
        self.delay = delay # in seconds
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def set_print_fn(self, fn):
        self.memory_manager.print_fn = fn

    def run(self):
        """Thread run code"""
        if self.delay:
            self.memory_manager.print("Starting bloating delay for {}s".format(self.delay))
            time.sleep(self.delay)
        self.memory_manager.print("Starting bloat")
        while True:
            self.memory_manager.refresh_allocation()
            self.memory_manager.print("Refreshed")
            time.sleep(self.refresh_interval)

    def percentage_available_system_memory(self):
        return self.memory_manager.percentage_available_system_memory()

    def percentage_used_manager_memory(self):
        return self.memory_manager.percentage_used_manager_memory()

if __name__ == "__main__":
    def test_manager():
        memory_manager = BloatingMemoryManager(0.8)
        print("(stale) free", memory_manager.percentage_available_system_memory())
        memory_manager.refresh_allocation()
        print("(fresh) free", memory_manager.percentage_available_system_memory())
        big_data = ' ' * 10**9
        print("(stale) free", memory_manager.percentage_available_system_memory())
        memory_manager.refresh_allocation()
        print("(fresh) free", memory_manager.percentage_available_system_memory())
        del big_data
        print("(stale) free", memory_manager.percentage_available_system_memory())
        memory_manager.refresh_allocation()
        print("(fresh) free", memory_manager.percentage_available_system_memory())
    def test_async_manager():
        memory_manager = AsyncBloatingMemoryManager(0.8, 0.20)
        time.sleep(1.0)
        print("(fresh) free", memory_manager.percentage_available_system_memory())
        big_data = ' ' * 10**9
        time.sleep(0.5)
        print("(fresh) free", memory_manager.percentage_available_system_memory())
        del big_data
        time.sleep(0.5)
        print("(fresh) free", memory_manager.percentage_available_system_memory())
        time.sleep(0.5)
        print("(fresh) free", memory_manager.percentage_available_system_memory())
        big_data = ' ' * 10**9
        time.sleep(2.0)
        print("(fresh) free", memory_manager.percentage_available_system_memory())
        del big_data
        time.sleep(2.0)
        print("(fresh) free", memory_manager.percentage_available_system_memory())
    test_manager()
    print("Async manager")
    test_async_manager()
    print("Sync manager")
    test_manager()
