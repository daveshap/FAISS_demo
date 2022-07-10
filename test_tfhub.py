import tensorflow as tf
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
embeddings = embed(["The quick brown fox jumps over the lazy dog.","I am a sentence for which I would like to get its embedding"])

print(embeddings)

# The following are example embedding output of 512 dimensions per sentence
# Embedding for: The quick brown fox jumps over the lazy dog.
# tf.Tensor([[0.01305108  0.02235125 -0.03263278, ...]], shape=(1, 512), dtype=float32)
# Embedding for: I am a sentence for which I would like to get its embedding.
# tf.Tensor([0.05833394 -0.0818501   0.06890938, ...]], shape=(1, 512), dtype=float32)
