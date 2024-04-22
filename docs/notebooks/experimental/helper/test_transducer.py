import unittest
import transducer

class TestGenerateActiveElements(unittest.TestCase):
    def test_generate_active_elements(self):
        # Test case 1
        number_scan_lines = 5
        transducer_number_elements = 10
        window_size = 5
        expected_output = [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        ]
        result = generate_active_elements(number_scan_lines, transducer_number_elements, window_size)
        print(f"Test case 1: number_scan_lines={number_scan_lines}, transducer_number_elements={transducer_number_elements}, window_size={window_size}")
        print("Generated output:")
        print(result)
        print("Expected output:")
        print(np.array(expected_output))
        self.assertTrue(np.array_equal(result, np.array(expected_output)))

        # Test case 2
        number_scan_lines = 3
        transducer_number_elements = 8
        window_size = 3
        expected_output = [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1]
        ]
        result = generate_active_elements(number_scan_lines, transducer_number_elements, window_size)
        print(f"\nTest case 2: number_scan_lines={number_scan_lines}, transducer_number_elements={transducer_number_elements}, window_size={window_size}")
        print("Generated output:")
        print(result)
        print("Expected output:")
        print(np.array(expected_output))
        self.assertTrue(np.array_equal(result, np.array(expected_output)))

    def test_invalid_inputs(self):
        # Test case 3: Invalid input - number_scan_lines <= 0
        with self.assertRaises(ValueError):
            generate_active_elements(number_scan_lines=0, transducer_number_elements=10, window_size=5)
            print("\nTest case 3: Caught expected ValueError for number_scan_lines <= 0")

        # Test case 4: Invalid input - transducer_number_elements <= 0
        with self.assertRaises(ValueError):
            generate_active_elements(number_scan_lines=5, transducer_number_elements=0, window_size=5)
            print("\nTest case 4: Caught expected ValueError for transducer_number_elements <= 0")

        # Test case 5: Invalid input - window_size <= 0
        with self.assertRaises(ValueError):
            generate_active_elements(number_scan_lines=5, transducer_number_elements=10, window_size=0)
            print("\nTest case 5: Caught expected ValueError for window_size <= 0")

        # Test case 6: Invalid input - window_size > transducer_number_elements
        with self.assertRaises(ValueError):
            generate_active_elements(number_scan_lines=5, transducer_number_elements=10, window_size=11)
            print("\nTest case 6: Caught expected ValueError for window_size > transducer_number_elements")

        # Test case 7: Invalid input - window_size is even
        with self.assertRaises(ValueError):
            generate_active_elements(number_scan_lines=5, transducer_number_elements=10, window_size=4)
            print("\nTest case 7: Caught expected ValueError for even window_size")

# Running the test directly
def run_tests():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGenerateActiveElements))
    runner = unittest.TextTestRunner()
    runner.run(suite)

run_tests()