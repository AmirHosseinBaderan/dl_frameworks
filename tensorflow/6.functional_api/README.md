# Step 6 — Functional API

## Goal
Build models with Keras' **Functional API**, which supports non-linear topologies
(branches, merges, multiple inputs/outputs) that `Sequential` cannot express.

## How it works
- Define an `Input` tensor: `inputs = tf.keras.Input(shape=(784,))`.
- Connect layers explicitly: `x = Dense(128, activation="relu")(inputs)`.
- Create the model: `model = tf.keras.Model(inputs=inputs, outputs=outputs)`.
- **Branches / merge**: two `Dense` branches concatenated with
  `tf.keras.layers.Concatenate()([branch1, branch2])`.
- **Multiple inputs**: e.g. an image input and a scalar input feeding into a merge.

Sequential is a straight line; Functional is a graph you draw with layer connections.

## Files
| File | Description |
|------|-------------|
| [`main.py`](main.py) | Builds models with the Functional API, including branching/concatenation and multi-input patterns, and prints summaries. |
