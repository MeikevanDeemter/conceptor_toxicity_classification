"""
This run script demonstrates how to use activation steering on simple examples.

Example usage for GPT-J-6B:
```bash
uv run python run_simple_demo.py \
    --examples=wedding \
    --model_name=EleutherAI/gpt-j-6b \
    --layers 15 16 \
    --aperture 0.01 0.1 \
    --beta 0 1 10
```

Example usage for GPT-2:
```bash
uv run python run_simple_demo.py \
    --examples=wedding \
    --model_name=gpt2 \
    --layers 8 9 \
    --aperture 0.01 0.1 \
    --beta 0 1 10
```
"""
import argparse
import json
import re
import random
import os
from functools import partial
from datetime import datetime
from typing import List
import itertools
from pathlib import Path

from hookedllm.HookedLLM import HookedLLM
from hookedllm.config import CachingConfig, ModuleSelection, TokenSelection
from hookedllm.config import MeanActivationConfig
from hookedllm.steering_functions import MeanActivationSteering
from hookedllm.config import ConceptorConfig
from hookedllm.steering_functions import ConceptorSteering
import torch


# ANSI escape codes for formatting
BOLD = '\033[1m'
RESET = '\033[0m'
BLUE = '\033[94m'

print("Hello, world!")
def get_examples(example_name: str) -> List[str]:
    positive_examples = [
        "I absolutely loved this movie because it was brilliant and entertaining throughout.",
        "The movie exceeded all my expectations with its masterful storytelling and direction.",
        "This film deserves an award for being so incredibly well-made and moving.",
        "Everyone should see this amazing movie at least once in their lifetime.",
        "The performances in this film were outstanding and truly captivating.",
        "I was completely blown away by how good this movie turned out to be.",
        "This is definitely one of the best films I've seen in recent years.",
        "The director created a stunning masterpiece that will stand the test of time.",
    ]
    love_examples = [
        "I love you with all my heart and soul.",
        "Love is the most powerful force in the universe.",
        "My heart beats only for you, my love.",
        "I adore everything about you, truly.",
        "Love conquers all obstacles in its path.",
        "You are the love of my life, forever and always.",
        "I cherish every moment we spend together.",
        "Love is patient, love is kind.",
        "My affection for you grows stronger each day.",
        "I'm completely devoted to you, my darling.",
        "Love illuminates even the darkest corners of life.",
        "You are my heart's greatest treasure.",
        "I'm endlessly in love with your beautiful soul.",
        "Love is the answer to life's hardest questions.",
        "My love for you knows no boundaries.",
        "I adore the way you light up my world.",
        "True love never fades, it only grows deeper.",
        "You are the love I've always dreamed of finding.",
        "I'm smitten by your charm and grace.",
        "Love fills my heart with indescribable joy.",
        "I worship the ground you walk on, my love.",
        "My affection for you is eternal and unchanging.",
        "Love is the most beautiful gift we can give.",
        "I'm head over heels in love with you.",
        "Your love is the greatest blessing in my life.",
        "I adore every little thing about you.",
        "Love transforms ordinary moments into magic.",
        "My heart belongs to you completely.",
        "I'm deeply in love with your mind and spirit.",
        "Love is what makes life worth living.",
        "I cherish our love story more than any other.",
        "My devotion to you is unwavering and true.",
        "Love shines brightest in times of darkness.",
        "I adore how you make me feel so alive.",
        "My affection for you grows with each passing day.",
        "Love is the bridge between two souls.",
        "I'm utterly captivated by your love.",
        "Your love is the compass that guides my heart.",
        "I treasure every smile, every touch, every kiss.",
        "Love is the greatest adventure of all.",
        "My heart sings whenever you're near me.",
        "I adore the way you understand me completely.",
        "Love is the most precious gift we can share.",
        "I'm completely enchanted by your loving nature.",
        "My affection for you is pure and unconditional.",
        "Love weaves our souls together eternally.",
        "I cherish the love we've built together.",
        "Your love is the light of my life.",
        "I adore you more than words can express.",
        "Love is all I need, and you are my everything."
    ]
    wedding_examples = [
        "Planning a wedding can be both exciting and challenging.",
        "Their best man was the talk of the town for weeks.",
        "A beautiful marriage is a celebration of love and commitment.",
        "The bride walked gracefully down the aisle to meet her groom.",
        "Wedding vows are personal promises exchanged during the ceremony.",
        "The reception venue was decorated with elegant floral arrangements.",
        "Choosing the perfect wedding dress took months of shopping.",
        "The bridal party helped with all the wedding preparations.",
        "They exchanged rings as a symbol of their eternal bond.",
        "The ceremony was officiated by a close family friend.",
        "Guests tossed confetti as the newlyweds exited the church.",
        "Their honeymoon in Bali was absolutely magical.",
        "The wedding cake had three tiers with buttercream frosting.",
        "Her bridesmaids wore matching gowns in pastel colors.",
        "Their engagement lasted exactly one year before the wedding.",
        "The rehearsal dinner was held at their favorite restaurant.",
        "They wrote their own vows to make the ceremony more personal.",
        "Wedding invitations were sent out three months in advance.",
        "The flower girl scattered rose petals along the aisle.",
        "Their first dance as husband and wife moved everyone to tears.",
        "Wedding photographers captured every special moment of the day.",
        "The bridal shower was organized by her maid of honor.",
        "A beautiful bouquet of white roses complemented her gown.",
        "The groom and his groomsmen wore matching tuxedos.",
        "Their wedding registry included items for their new home.",
        "The matrimony was witnessed by their closest family and friends.",
        "Catering for the reception featured a five-course gourmet meal.",
        "Save-the-date cards were designed with their engagement photo.",
        "Wedding guests traveled from across the country to attend.",
        "The couple exchanged their vows at sunset by the ocean.",
        "Bridesmaids helped the bride prepare on the morning of the wedding.",
        "The cathedral was adorned with ribbons and fresh flowers.",
        "Their matrimonial celebration continued well into the night.",
        "Wedding bands were custom designed to match her engagement ring.",
        "The newlyweds departed amid a shower of rice and well-wishes.",
        "Her veil was handmade using lace from her mother's wedding dress.",
        "The wedding procession entered to classical string quartet music.",
        "Table centerpieces featured candles and seasonal blooms.",
        "Marriage licenses must be obtained before the ceremony.",
        "The wedding planner ensured everything ran smoothly.",
        "Toasts from the best man and maid of honor were heartfelt.",
        "The bridal bouquet was tossed to eager single ladies.",
        "Their union was blessed by both families during the ceremony.",
        "A string quartet played during the cocktail hour.",
        "Wedding favors were personalized with the couple's initials.",
        "The bride's father gave a moving speech at the reception.",
        "Pre-wedding jitters disappeared as soon as they saw each other.",
        "After exchanging vows, they were pronounced husband and wife.",
        "Their anniversary will always remind them of their special day.",
        "The newlyweds sealed their marriage with a romantic kiss."
    ]
    return {
        "wedding": wedding_examples,
        "love": love_examples,
        "positive-movie": positive_examples,
    }[example_name]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt2",
        help="The name of the model to use"
    )
    parser.add_argument(
        "--layer_path",
        type=str,
        nargs='?',
        help="Path to the module list to be steered"
    )
    parser.add_argument(
        "--layers", 
        type=int,
        nargs='*',
        help="The layers in the module list to use"
    )
    parser.add_argument(
        "--layer_modules", 
        type=str, 
        nargs='*',
        help="The layers in the module list to use"
    )   
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seed to use"
    )
    parser.add_argument(
        "--examples",
        type=str,
        choices=["wedding", "love", "positive-movie"],
        default="wedding",
        help="The examples to use"
    )
    parser.add_argument(
        "--cache_position",
        type=str,
        default=":5",
        help="* for all tokens, :N for last N tokens"
    )
    parser.add_argument(
        "--beta",
        type=float,
        nargs='+',
        default=[0.0, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0],
        help="The beta values to use"
    )
    parser.add_argument(
        "--steering_function",
        type=str,
        choices=["activation", "conceptor"],
        default="conceptor",
        help="The steering function to use"
    )
    # conceptor-specific args
    parser.add_argument(
        "--aperture",
        type=float,
        nargs='+',
        default=[0.01],
        help="The aperture values to use"
    )
    parser.add_argument(
        "--additive_conceptor",
        action="store_true",
        help="Whether to use additive conceptor steering"
    )
    
    args = parser.parse_args()

    # default args for tested models
    defaults = {
        'gpt2': {
            'layer_path': 'transformer.h',
            'layers': [8, 9],
            'layer_modules': ['ln_1']
        },
        'EleutherAI/gpt-j-6b': {
            'layer_path': 'transformers.h',
            'layers': [15, 16],
            'layer_modules': ['ln_1']
        },
        'google/recurrentgemma-2b': {
            'layer_path': 'model.layers',
            'layers': [13],
            'layer_modules': ['temporal_pre_norm']
        },
        'ai21labs/AI21-Jamba-Mini-1.6': {
            'layer_path': 'model.layers',
            'layers': [19],
            'layer_modules': ['pre_ff_layernorm']
        },
        'state-spaces/mamba-130m-hf': {
            'layer_path': 'backbone.layers',
            'layers': [16],
            'layer_modules': ['norm']
        },
        'AntonV/mamba2-2.7b-hf': {
            'layer_path': 'backbone.layers',
            'layers': [41, 42],
            'layer_modules': ['norm']
        }
    }
    # Ensure module specification
    for arg in ['layer_path', 'layers', 'layer_modules']: 
        if not getattr(args, arg):
            setattr(args, arg, defaults[args.model_name][arg])

    # Parse cache position
    if args.cache_position == "*":
        args.cache_position = None
    elif args.cache_position.startswith(":"):
        args.cache_position = list(range(-int(args.cache_position[1:]), 0))
    else:
        raise ValueError(f"Invalid cache position: {args.cache_position}")

    return args


def print_args(args):
    print(f"Running with args:")
    print('\n'.join([f'  {k}: {v}' for k, v in vars(args).items()]), end='\n\n')


def format_generation(prompt, generated_text):
    text = re.sub(r'\s+', ' ', generated_text)
    n_prompt = len(prompt)
    highlighted_text = f"{BLUE}{text[:n_prompt]}{RESET}{text[n_prompt:]}"
    return highlighted_text


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Print args nicely
    print_args(args)

    # Initialize model and module selection
    model = HookedLLM(args.model_name, debug_mode=args.debug)
    module_selection = ModuleSelection(
        layers=args.layers,
        layer_modules=args.layer_modules,
        layer_path=args.layer_path,
    )

    # (1) Extract activations for a specific concept
    caching_config = CachingConfig(
        modules=module_selection,
        tokens=TokenSelection(position=args.cache_position),
    )

    # Get dataset of examples
    examples: List[str] = get_examples(args.examples)
    # Use the same prompts as the optimization script for apples-to-apples comparison
    # TODO: Make this a parameter
    prompts = [
        "I went to the event and it was",
        "The ceremony felt very",
        "Let's talk about",
        "The best part of the day was",
        "After the party, they went on their",
        # "This movie was ",
        # "I thought the film was ",
        # "After watching it, I felt "
    ]

    # Collect activations for positive examples
    print("Collecting activations for examples...")
    with model.caching(caching_config):
        # TODO: process examples in batches
        for example in examples:
            model.forward(example)

    # Get the cached activations
    act = model.get_and_clear_cache()
    act = {
        k: torch.cat(v, dim=0)
        for k, v in act.items()
    }
    print(f"Collected activations for {len(examples)} examples.")

    # Choose steering function
    if args.steering_function == "activation":
        hyperparam_names = ["beta"]
        hyperparam_values = [args.beta]
        cfg_cls = MeanActivationConfig
        steering_fn_cls = MeanActivationSteering
    elif args.steering_function == "conceptor":
        hyperparam_names = ["beta", "aperture"]
        hyperparam_values = [args.beta, args.aperture]
        cfg_cls = partial(ConceptorConfig, additive=args.additive_conceptor)
        steering_fn_cls = ConceptorSteering
    else:
        raise ValueError(f"Invalid steering function: {args.steering_function}")

    # product of all hyperparams so we can iterate over them
    hyperparam_combinations = list(itertools.product(*hyperparam_values))
    results = {}
    for hyperparam_tuple in hyperparam_combinations:
        # create dict mapping hyperparameter names to values
        hyperparam_dict = dict(zip(hyperparam_names, hyperparam_tuple))
        hyperparam_strl = [f"{k}={v}" for k, v in hyperparam_dict.items()]
        print(f"\n----- Running with {', '.join(hyperparam_strl)} -----")

        # (3) Create steering function
        # TODO: can also compute steering fn once and change hyperparams (more efficient)
        steering_config = cfg_cls(
            modules=module_selection,
            tokens=TokenSelection(),  # For generation, steer all tokens
            **hyperparam_dict,
        )
        steering_fn = steering_fn_cls(
            config=steering_config,
            activations=act,
        )

        # TODO: process prompts in batches
        for idx, prompt in enumerate(prompts):
            with model.steering(steering_fn, steering_config):
                output = model.generate(
                    prompt,
                    generation_kwargs={
                        "max_length": 30,
                        "do_sample": True,
                        "temperature": 0.8,  # Slightly higher temperature for more variety
                        "top_p": 0.95,  # Higher top_p for more diversity
                        "no_repeat_ngram_size": 3,  # Avoid repetition
                        "pad_token_id": model.tokenizer.eos_token_id,
                    }
                )
            generated_text = model.tokenizer.decode(output[0], skip_special_tokens=True)

            # Show clear output with prompt vs. generated text
            print_text = format_generation(prompt, generated_text)
            print(f"Prompt {idx+1}: {print_text}")

            # store results
            results[f"{'_'.join(hyperparam_strl)}_{idx}"] = {
                "prompt": prompt,
                "generated_text": generated_text,
            }

    # Save results to results folder directory in the root of repository
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parents[1] / "results" / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    with open(results_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
