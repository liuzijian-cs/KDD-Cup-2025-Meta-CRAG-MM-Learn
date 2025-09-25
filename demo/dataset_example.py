from datasets import load_dataset

# --- load datasets ---------------------------------------------------------
st = load_dataset("crag-mm-2025/crag-mm-single-turn-public",
                  split="validation", revision="v0.1.2")
mt = load_dataset("crag-mm-2025/crag-mm-multi-turn-public",
                  split="validation", revision="v0.1.2")

# --- iterate over turns, schema-proof --------------------------------------
def iter_turns(sample):
    """Yield (turn_dict, answer_dict) pairs in either single- or multi-turn rows."""
    if isinstance(sample["turns"], dict):
        n = len(sample["turns"]["interaction_id"])
        for i in range(n):
            turn =  {k: v[i] for k, v in sample["turns"].items()}
            ans  =  {k: v[i] for k, v in sample["answers"].items()}
            yield turn, ans
    else:  # older releases only
        for turn, ans in zip(sample["turns"], sample["answers"]):
            yield turn, ans

# inspect first multi-turn conversation
for t, a in iter_turns(st[0]):
    print(f"Q: {t['query']}\nA: {a['ans_full']}\n")

# show the (possibly down-sampled) image
import matplotlib.pyplot as plt
plt.imshow(st[0]["image"])
plt.axis("off")
plt.show()