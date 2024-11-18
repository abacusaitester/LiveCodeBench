import unittest
import argparse
from unittest.mock import patch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

class TestParser(unittest.TestCase):

    def test_default_arguments(self):
        with patch('sys.argv', ['script']):
            args = get_args()
            self.assertEqual(args.model, "gpt-3.5-turbo-0301")
            self.assertEqual(args.scenario, Scenario.codegeneration)
            self.assertEqual(args.n, 10)
            self.assertEqual(args.temperature, 0.2)
            self.assertEqual(args.top_p, 0.95)
            self.assertEqual(args.max_tokens, 2000)
            self.assertEqual(args.stop, ["###"])
            self.assertFalse(args.debug)
            self.assertFalse(args.evaluate)

    def test_custom_arguments(self):
        test_args = [
            'script',
            '--model', 'gpt-4',
            '--scenario', 'codeexecution',
            '--n', '20',
            '--temperature', '0.5',
            '--top_p', '0.9',
            '--max_tokens', '1000',
            '--stop', 'END,STOP',
            '--debug',
            '--evaluate'
        ]
        with patch('sys.argv', test_args):
            args = get_args()
            self.assertEqual(args.model, "gpt-4")
            self.assertEqual(args.scenario, Scenario.codeexecution)
            self.assertEqual(args.n, 20)
            self.assertEqual(args.temperature, 0.5)
            self.assertEqual(args.top_p, 0.9)
            self.assertEqual(args.max_tokens, 1000)
            self.assertEqual(args.stop, ["END", "STOP"])
            self.assertTrue(args.debug)
            self.assertTrue(args.evaluate)

    def test_tensor_parallel_size(self):
        with patch('sys.argv', ['script']):
            with patch('torch.cuda.device_count', return_value=4):
                args = get_args()
                self.assertEqual(args.tensor_parallel_size, 4)

        with patch('sys.argv', ['script', '--tensor_parallel_size', '2']):
            args = get_args()
            self.assertEqual(args.tensor_parallel_size, 2)

    def test_multiprocess(self):
        with patch('sys.argv', ['script']):
            with patch('os.cpu_count', return_value=8):
                args = get_args()
                self.assertEqual(args.multiprocess, 0)  # Default value

        with patch('sys.argv', ['script', '--multiprocess', '-1']):
            with patch('os.cpu_count', return_value=8):
                args = get_args()
                self.assertEqual(args.multiprocess, 8)

    def test_invalid_scenario(self):
        with patch('sys.argv', ['script', '--scenario', 'invalid_scenario']):
            with self.assertRaises(SystemExit):
                get_args()

    def test_boolean_flags(self):
        test_args = [
            'script',
            '--trust_remote_code',
            '--not_fast',
            '--cot_code_execution',
            '--continue_existing',
            '--continue_existing_with_eval',
            '--use_cache',
            '--enable_prefix_caching'
        ]
        with patch('sys.argv', test_args):
            args = get_args()
            self.assertTrue(args.trust_remote_code)
            self.assertTrue(args.not_fast)
            self.assertTrue(args.cot_code_execution)
            self.assertTrue(args.continue_existing)
            self.assertTrue(args.continue_existing_with_eval)
            self.assertTrue(args.use_cache)
            self.assertTrue(args.enable_prefix_caching)

if __name__ == '__main__':
    unittest.main()
