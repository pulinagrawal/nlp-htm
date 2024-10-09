import unittest
from utils.htm_helpers import HTMModel

class TestHTMModel(unittest.TestCase):
    def test_get_compute_order(self):
        htm_model = HTMModel()
        htm_model.add_link("region1", "region3", "BU")
        htm_model.add_link("region2", "region3", "BU")
        htm_model.add_link("region3", "region4", "BU")
        htm_model.add_link("region4", "region5", "BU")
        htm_model.add_link("region5", "region6", "BU")
    
        expected_order = ["region1", "region2", "region3", "region4", "region5", "region6"]
        actual_order = htm_model._get_compute_order()
    
        self.assertEqual(actual_order, expected_order)

        # Test with different order
        htm_model = HTMModel()
        htm_model.add_link("region2", "region3", "BU")
        htm_model.add_link("region1", "region3", "BU")
        htm_model.add_link("region3", "region4", "BU")
        htm_model.add_link("region4", "region5", "BU")
        htm_model.add_link("region5", "region6", "BU")
    
        actual_order = htm_model._get_compute_order()
        self.assertNotEqual(actual_order, expected_order)

        expected_order = ["region2", "region1", "region3", "region4", "region5", "region6"]
        self.assertEqual(actual_order, expected_order)

        # Test assymetric heirarchy input links
        htm_model = HTMModel()
        htm_model.add_link("region1", "region3", "BU")
        htm_model.add_link("region2", "region3", "BU")
        htm_model.add_link("region3", "region5", "BU")
        htm_model.add_link("region4", "region5", "BU")
        htm_model.add_link("region5", "region6", "BU")
    
        expected_order = ["region1", "region2", "region4", "region3", "region5", "region6"]
        actual_order = htm_model._get_compute_order()
    
        self.assertEqual(actual_order, expected_order)

    def test_region_pairs(self):
        htm_model = HTMModel()
        htm_model.add_link("region1", "region2", "BU")
        htm_model.add_link("region1", "region3", "BU")
        htm_model.add_link("region2", "region3", "BU")
        htm_model.add_link("region3", "region4", "BU")
        htm_model.add_link("region4", "region5", "BU")
        
        expected_pairs = set([("region1", "region2"),
                          ("region2", "region3"),
                          ("region1", "region3"),
                          ("region3", "region4"),
                          ("region4", "region5")])
        actual_pairs = set(htm_model.region_pairs())
        
        self.assertEqual(actual_pairs, expected_pairs)

if __name__ == '__main__':
    unittest.main()
    # ... existing test cases ...
