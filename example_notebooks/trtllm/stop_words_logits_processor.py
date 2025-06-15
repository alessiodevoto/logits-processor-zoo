from transformers import AutoTokenizer
from logits_processor_zoo.trtllm import StopWordsLogitsProcessor
from utils import TRTLLMTester, get_parser


if __name__ == "__main__":
    args = get_parser()
    beam_width = 1

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Define stop words that should trigger generation to stop
    stop_words = ["Thank you", "In conclusion", "To summarize"]
    
    # Create the logits processor with a high boost factor to ensure stopping
    lp = StopWordsLogitsProcessor(tokenizer, stop_words, boost_factor=100.0)

    # Run generation with the stop words processor
    TRTLLMTester(lp, tokenizer, args).run(args.prompt, beam_width) 
