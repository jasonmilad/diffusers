from diffusers import DiffusionPipeline
import torch
import numpy as np


model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    dtype = torch.bfloat16
else:
    dtype = torch.float32

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype, device_map="balanced")

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

negative_prompt = " " # Recommended if you don't use a negative prompt.


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (512, 512),
}

width, height = aspect_ratios["1:1"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]


# Extract captured queries
block_0_queries = pipe.transformer.transformer_blocks[0].attn._captured_queries
# block_59_queries = pipe.transformer.transformer_blocks[59].attn._captured_queries

def analyze_consecutive_similarity(queries):
    """
    queries: tensor of shape [batch, seq_len, num_heads, head_dim]
    """
    batch, seq_len, num_heads, head_dim = queries.shape
    # Flatten heads
    queries_flat = queries.reshape(batch, seq_len, num_heads * head_dim)
    
    similarities = []
    for b in range(batch):
        for i in range(seq_len - 1):
            q1 = queries_flat[b, i]
            q2 = queries_flat[b, i+1]
            sim = torch.cosine_similarity(q1.unsqueeze(0), q2.unsqueeze(0))
            similarities.append(sim.item())
    
    hist, bin_edges = np.histogram(similarities, bins=20)
    
    return {
        'mean': np.mean(similarities),
        'std': np.std(similarities),
        'min': np.min(similarities),
        'max': np.max(similarities),
        'median': np.median(similarities),
        'hist': hist,
        'bin_edges': bin_edges,
        'similarities': similarities  # Keep for further analysis if needed
    }

def print_stats(block_name, timestep_idx, stats):
    print(f"\n{'='*60}")
    print(f"{block_name} - Timestep {timestep_idx}")
    print(f"{'='*60}")
    print(f"Mean similarity:   {stats['mean']:.4f}")
    print(f"Median similarity: {stats['median']:.4f}")
    print(f"Std deviation:     {stats['std']:.4f}")
    print(f"Min similarity:    {stats['min']:.4f}")
    print(f"Max similarity:    {stats['max']:.4f}")
    
    # Print histogram
    print(f"\nDistribution of similarities:")
    print(f"{'Range':<20} {'Count':<10} {'Bar'}")
    print(f"{'-'*50}")
    
    max_count = max(stats['hist'])
    for count, (left, right) in zip(stats['hist'], zip(stats['bin_edges'][:-1], stats['bin_edges'][1:])):
        bar_length = int(30 * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_length
        print(f"{left:.3f} - {right:.3f}    {count:<10} {bar}")

# Analyze and print
print("\n" + "="*60)
print("QUERY SIMILARITY ANALYSIS")
print("="*60)

for timestep_idx, captured in enumerate(block_0_queries):
    stats = analyze_consecutive_similarity(captured['joint'])
    timestep_name = timestep_idx
    # print_stats(f"Block 0 ({timestep_name})", timestep_idx, stats)

# for timestep_idx, captured in enumerate(block_59_queries):
    # stats = analyze_consecutive_similarity(captured['joint'])
    # timestep_name = timestep_idx
    # print_stats(f"Block 59 ({timestep_name})", timestep_idx, stats)

# Summary comparison
print(f"\n{'='*60}")
print("SUMMARY COMPARISON")
print(f"{'='*60}")
print(f"{'Block':<15} {'Timestep':<15} {'Mean':<10} {'Std':<10}")
print(f"{'-'*60}")

for timestep_idx, captured in enumerate(block_0_queries):
    stats = analyze_consecutive_similarity(captured['joint'])
    ts_name = captured['timestep']
    print(f"{'Block 0':<15} {ts_name:<15} {stats['mean']:.4f}    {stats['std']:.4f}")

# for timestep_idx, captured in enumerate(block_59_queries):
#     stats = analyze_consecutive_similarity(captured['joint'])
#     ts_name = timestep_idx
#     print(f"{'Block 59':<15} {ts_name:<15} {stats['mean']:.4f}    {stats['std']:.4f}")


image.save("example.png")