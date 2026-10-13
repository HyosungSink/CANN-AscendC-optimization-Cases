"""Prevent out-of-scope or ambiguous NPU names from reaching the kernels."""
import unittest

from device_guard import require_a2_a3


class DeviceGuardTest(unittest.TestCase):
    def test_known_a2_a3_families(self):
        for name in ("Ascend910B", "Ascend910B1", "Ascend910B2C",
                     "Ascend910_9382", "Ascend910_9391", "Ascend910C"):
            with self.subTest(name=name):
                require_a2_a3(name)

    def test_out_of_scope_and_unknown_names(self):
        for name in ("Ascend310P", "Ascend950", "Ascend950PR_9599",
                     "Ascend910", "Ascend910A", "Ascend910_9599",
                     "unknown", "", "Ascend910B_unknown"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    require_a2_a3(name)


if __name__ == "__main__":
    unittest.main()
