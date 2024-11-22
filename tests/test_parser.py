import unittest
import argparse
from unittest.mock import patch
import torch
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

class TestParser(unittest.TestCase):

    def test_default_values(self):
        with patch('sys.argv', ['script']):
            args = get_args()
            self.assertEqual(args.model, "gpt-3.5-turbo-0301")
            self.assertIsNone(args.local_model_path)
            self.assertFalse(args.trust_remote_code)
            self.assertEqual(args.scenario, Scenario.codegeneration)
            self.assertFalse(args.not_fast)
            self.assertEqual(args.release_version, "release_v1")
            self.assertFalse(args.cot_code_execution)
            self.assertEqual(args.n, 10)
            self.assertEqual(args.codegen_n, 10)
            self.assertEqual(args.temperature, 0.2)
            self.assertEqual(args.top_p, 0.95)
            self.assertEqual(args.max_tokens, 2000)
            self.assertEqual(args.multiprocess, 0)
            self.assertEqual(args.stop, ["###"])
            self.assertFalse(args.continue_existing)
            self.assertFalse(args.continue_existing_with_eval)
            self.assertFalse(args.use_cache)
            self.assertEqual(args.cache_batch_size, 100)
            self.assertFalse(args.debug)
            self.assertFalse(args.evaluate)
            self.assertEqual(args.num_process_evaluate, 12)
            self.assertEqual(args.timeout, 6)
            self.assertEqual(args.openai_timeout, 45)
            self.assertEqual(args.tensor_parallel_size, torch.cuda.device_count())
            self.assertFalse(args.enable_prefix_caching)
            self.assertIsNone(args.custom_output_file)
            self.assertIsNone(args.custom_output_save_name)
            self.assertEqual(args.dtype, "bfloat16")

    def test_custom_values(self):
        test_args = [
            'script',
            '--model', 'custom-model',
            '--local_model_path', '/path/to/model',
            '--trust_remote_code',
            '--scenario', 'codeexecution',
            '--not_fast',
            '--release_version', 'v2',
            '--cot_code_execution',
            '--n', '20',
            '--codegen_n', '15',
            '--temperature', '0.5',
            '--top_p', '0.9',
            '--max_tokens', '3000',
            '--multiprocess', '4',
            '--stop', '###,END',
            '--continue_existing',
            '--use_cache',
            '--cache_batch_size', '200',
            '--debug',
            '--evaluate',
            '--num_process_evaluate', '8',
            '--timeout', '10',
            '--openai_timeout', '60',
            '--tensor_parallel_size', '2',
            '--enable_prefix_caching',
            '--custom_output_file', 'output.txt',
            '--custom_output_save_name', 'results',
            '--dtype', 'float16'
        ]

        with patch('sys.argv', test_args):
            args = get_args()
            self.assertEqual(args.model, "custom-model")
            self.assertEqual(args.local_model_path, "/path/to/model")
            self.assertTrue(args.trust_remote_code)
            self.assertEqual(args.scenario, Scenario.codeexecution)
            self.assertTrue(args.not_fast)
            self.assertEqual(args.release_version, "v2")
            self.assertTrue(args.cot_code_execution)
            self.assertEqual(args.n, 20)
            self.assertEqual(args.codegen_n, 15)
            self.assertEqual(args.temperature, 0.5)
            self.assertEqual(args.top_p, 0.9)
            self.assertEqual(args.max_tokens, 3000)
            self.assertEqual(args.multiprocess, 4)
            self.assertEqual(args.stop, ["###", "END"])
            self.assertTrue(args.continue_existing)
            self.assertTrue(args.use_cache)
            self.assertEqual(args.cache_batch_size, 200)
            self.assertTrue(args.debug)
            self.assertTrue(args.evaluate)
            self.assertEqual(args.num_process_evaluate, 8)
            self.assertEqual(args.timeout, 10)
            self.assertEqual(args.openai_timeout, 60)
            self.assertEqual(args.tensor_parallel_size, 2)
            self.assertTrue(args.enable_prefix_caching)
            self.assertEqual(args.custom_output_file, "output.txt")
            self.assertEqual(args.custom_output_save_name, "results")
            self.assertEqual(args.dtype, "float16")

if __name__ == '__main__':
    unittest.main()
