import unittest
import argparse
from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario

class TestParser(unittest.TestCase):
    def test_default_args(self):
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
        self.assertEqual(args.tensor_parallel_size, 1)  # Assuming 1 CUDA device
        self.assertFalse(args.enable_prefix_caching)
        self.assertIsNone(args.custom_output_file)
        self.assertIsNone(args.custom_output_save_name)
        self.assertEqual(args.dtype, "bfloat16")

    def test_custom_args(self):
        test_args = [
            "--model", "gpt-4",
            "--local_model_path", "/path/to/model",
            "--trust_remote_code",
            "--scenario", "selfrepair",
            "--not_fast",
            "--release_version", "release_v2",
            "--cot_code_execution",
            "--n", "20",
            "--codegen_n", "15",
            "--temperature", "0.5",
            "--top_p", "0.9",
            "--max_tokens", "3000",
            "--multiprocess", "4",
            "--stop", "END,STOP",
            "--continue_existing",
            "--use_cache",
            "--cache_batch_size", "200",
            "--debug",
            "--evaluate",
            "--num_process_evaluate", "8",
            "--timeout", "10",
            "--openai_timeout", "60",
            "--tensor_parallel_size", "2",
            "--enable_prefix_caching",
            "--custom_output_file", "/path/to/output",
            "--custom_output_save_name", "custom_results",
            "--dtype", "float16"
        ]
        
        args = get_args()
        parsed_args = args.parse_args(test_args)
        
        self.assertEqual(parsed_args.model, "gpt-4")
        self.assertEqual(parsed_args.local_model_path, "/path/to/model")
        self.assertTrue(parsed_args.trust_remote_code)
        self.assertEqual(parsed_args.scenario, Scenario.selfrepair)
        self.assertTrue(parsed_args.not_fast)
        self.assertEqual(parsed_args.release_version, "release_v2")
        self.assertTrue(parsed_args.cot_code_execution)
        self.assertEqual(parsed_args.n, 20)
        self.assertEqual(parsed_args.codegen_n, 15)
        self.assertEqual(parsed_args.temperature, 0.5)
        self.assertEqual(parsed_args.top_p, 0.9)
        self.assertEqual(parsed_args.max_tokens, 3000)
        self.assertEqual(parsed_args.multiprocess, 4)
        self.assertEqual(parsed_args.stop, ["END", "STOP"])
        self.assertTrue(parsed_args.continue_existing)
        self.assertTrue(parsed_args.use_cache)
        self.assertEqual(parsed_args.cache_batch_size, 200)
        self.assertTrue(parsed_args.debug)
        self.assertTrue(parsed_args.evaluate)
        self.assertEqual(parsed_args.num_process_evaluate, 8)
        self.assertEqual(parsed_args.timeout, 10)
        self.assertEqual(parsed_args.openai_timeout, 60)
        self.assertEqual(parsed_args.tensor_parallel_size, 2)
        self.assertTrue(parsed_args.enable_prefix_caching)
        self.assertEqual(parsed_args.custom_output_file, "/path/to/output")
        self.assertEqual(parsed_args.custom_output_save_name, "custom_results")
        self.assertEqual(parsed_args.dtype, "float16")

if __name__ == '__main__':
    unittest.main()
