# CNNs explained, and how this project was built

A guide for beginners and intermediate readers. Part 1 explains the ideas behind CNNs. Part 2 explains how they are coded from scratch. Part 3 walks through this project step by step. Part 4 gives the results and what they teach.

---

## Part 1. CNN basics

### 1.1 What is a CNN?

A **convolutional neural network (CNN)** is a neural network designed for images. It slides small **filters** over the image to detect simple patterns (strokes, corners), then combines those into larger ones (wheels, a ladder, a cab). It learns the filters by itself from labelled examples.

Why not a plain **dense** network, where every input connects to every neuron? Flattening an image throws away which pixels are neighbours, and a dense layer must learn the same pattern separately for every position. A CNN reuses one filter at every position (**weight sharing**), so it needs far fewer parameters and finds a pattern wherever it appears. In this project, a dense network scored about 70% and a CNN about 78% on identical data.

### 1.2 An image is a grid of numbers

Our drawings are 28×28 grayscale grids. Each pixel is a number from 0 to 255 (strokes are high values, the background is 0). We scale them to 0–1 and arrange a batch as **(batch, channels, height, width)**. For 64 drawings that is (64, 1, 28, 28). "Channels" means values per pixel: 1 for grayscale, 3 for colour.

### 1.3 Convolution (the core idea)

A **filter** (or kernel) is a small grid of weights, here 3×3. Put it on a 3×3 patch of the image, multiply matching numbers, add them up and add a **bias** (one extra learned number). That gives one number. Slide the filter over every position and you get a grid of numbers called a **feature map**. It shows where the filter's pattern appears.

```
filter (vertical line)    patch with a line      empty patch
0 1 0                     0 9 0                  0 0 0
0 1 0                     0 9 0                  0 0 0
0 1 0                     0 9 0                  0 0 0
                          sum = 9+9+9 = 27       sum = 0
```

The filter responds strongly to a vertical stroke and not at all to an empty patch. Real filters are not designed by hand. Training finds useful ones.

- **Many filters, many channels.** Conv layer 1 has 16 filters, so it outputs 16 feature maps, stacked as 16 **channels**.
- **3D filters.** When the input already has several channels (conv layer 2 receives 16), each filter is 3D: 16×3×3. It sums over all input channels. Conv layer 2 has 32 such filters, so it has 32 × 16 × 3 × 3 + 32 biases = 4,640 parameters.
- **Padding and stride.** *Padding* adds a border of zeros so edge pixels also get filtered. *Stride* is the step size of the slide. The output size is **(H + 2·pad − k) / stride + 1**. With k=3, pad=1, stride=1 the size stays 28.

### 1.4 ReLU

**ReLU** is the function max(0, x): negative values become 0, positive values pass through. Without it, stacked layers would collapse into one big linear function, so it is what lets the network learn complex shapes. In the backward pass the gradient flows only where the input was positive.

### 1.5 Max pooling

A **max pool** layer takes each 2×2 window and keeps only the largest value, which halves the height and width. It keeps the strongest response, makes the network tolerant to small shifts, and cuts computation. It has no learnable parameters. In the backward pass only the winning position receives the gradient.

```
1 3
2 0   ->  3
```

### 1.6 Flatten and dense layers

After the convolution and pooling stages we have 32 feature maps of 7×7. **Flatten** unrolls them into one vector of 32 × 7 × 7 = 1,568 numbers. A **dense layer** computes y = xW + b and combines all of those features into scores for each class.

### 1.7 Softmax and cross-entropy

The last dense layer outputs 10 raw scores called **logits**. **Softmax** turns them into probabilities that add up to 1. **Cross-entropy** loss is −log(probability given to the correct class): small when the network is confident and right, large when it is confident and wrong. At a random start, the loss is ln(10) ≈ 2.30. Softmax and cross-entropy together have a very simple gradient: (probabilities − one-hot label) / batch size. For numerical stability we subtract the largest logit before exponentiating.

### 1.8 Receptive field

The **receptive field** is the part of the original image that one unit "sees". It grows with depth: after conv 1 it is 3×3 pixels, after the pool 4×4, after conv 2 8×8, and after the second pool 10×10. Early layers see strokes, later layers see parts of shapes.

### 1.9 How the network learns

1. **Forward pass:** the image goes through the layers to a prediction, and the loss is computed.
2. **Backward pass (backpropagation):** using the chain rule, each layer receives the gradient from the layer above and passes one down to the layer below. The **gradient** says how much the loss changes if a weight changes. The backward pass only *computes* gradients and changes nothing.
3. **SGD update:** each weight moves a small step against its gradient: w ← w − lr × gradient.

Key terms:
- **Mini-batch:** the update happens after each batch of 64 images, not after the whole dataset. This is faster, and the noise does no harm.
- **Epoch:** one pass through all training images (37,500 / 64 ≈ 586 updates).
- **Learning rate (lr):** the step size. Too small is slow, too large overshoots. We used 0.05.
- **He initialisation:** weights start random with standard deviation √(2 / fan-in), where fan-in is the number of inputs per neuron. This keeps the signal size steady through ReLU layers. Biases start at 0.
- **Hyperparameters** (lr, batch size, epochs, architecture) are chosen by us and not learned. We tuned them starting from random weights. *Fine-tuning*, meaning starting from a pretrained network, was not used.

### 1.10 Overfitting, and train / validation / test

**Overfitting** means the network memorises the training images instead of learning general patterns. The sign is training accuracy rising while validation accuracy stalls or falls, which is called the **train–validation gap**.

That's why the data is split three ways:
- **Train:** used to update the weights.
- **Validation:** used to make decisions (which epoch, which architecture).
- **Test:** used once, at the end, to report the final number. Once decisions have been influenced by the test set, it no longer gives an honest score.

Because the validation set is finite, its readings are noisy: about ±1 percentage point at ~6,000 images. Differences smaller than that are not evidence. A 95% margin for an accuracy p on n images is about ±1.96 × √(p(1−p)/n).

### 1.11 Data augmentation

**Augmentation** shows the network slightly altered copies of the training images. It acts as a regulariser, because the network can no longer memorise exact pixel positions. Here we used random horizontal flips (a vehicle facing left or right is still the same class) and random shifts of up to 2 pixels, on training batches only.

---

## Part 2. How a CNN is coded from scratch

- **im2col.** A direct convolution needs slow nested loops. im2col rewrites every 3×3 patch as one row of a big matrix, so the whole convolution becomes a single matrix multiplication, which NumPy runs fast. The price is memory: the patch matrix is about 9 times larger per input channel than the image. **col2im** is the reverse for gradients: it scatters rows back to pixels and *adds* overlapping contributions (a scatter-add).
- **Gradient checking.** Every backward pass written by hand is compared against a numerical estimate (nudge a weight slightly, measure how the loss changes). If they agree, the hand-written gradients are right. This is the project's main safety net.
- **Other tests.** Hand-worked examples, shape checks, and a comparison of the fast convolution against slow explicit loops. In total, 22 tests (`python -m pytest tests -v`).
- **Layer design.** Each layer has `forward`, `backward` and `params()` (the list of weights and their gradients). `Sequential` chains layers, and `SGD` updates whatever `params()` returns.

---

## Part 3. The project, step by step

1. **Goal and rules.** Build a CNN in pure NumPy with every forward and backward pass written by hand: no PyTorch, TensorFlow, autograd or scikit-learn. Speed comes from NumPy and im2col, and the GPU was never used.
2. **Data.** Quick, Draw! (Google, CC BY 4.0): 28×28 bitmaps of doodles. We chose 10 vehicle classes: train, tractor, speedboat, car, motorbike, firetruck, bulldozer, ambulance, airplane, helicopter. The raw files hold 117k to 221k drawings per class.
3. **Preprocessing** (`preprocess.py`). The same number of drawings per class (the raw classes are imbalanced), split 75 / 12.5 / 12.5 per class, so every split is balanced. With 5,000 per class that is 37,500 train, 6,250 validation and 6,250 test images. Data is stored as uint8 and scaled to 0–1 per batch. Early development used a smaller 3,000-per-class set.
4. **Building blocks.** ReLU, Dense (He init), fused softmax cross-entropy, each checked with gradient tests.
5. **Baseline.** A dense network (784 → 256 → ReLU → 10) trained with SGD, lr 0.05. About 67–69% validation accuracy. This is the number the CNN had to beat. A higher learning rate only made it memorise faster, so the ceiling stayed near 67–70%.
6. **Convolution, pooling, flatten.** Conv2D with im2col/col2im, MaxPool2D, Flatten, all gradient-checked, including the full chain Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Dense → loss.
7. **CNN architecture.**
   ```
   (1,28,28) -> Conv 3x3, 16 filters -> ReLU -> MaxPool -> (16,14,14)
             -> Conv 3x3, 32 filters -> ReLU -> MaxPool -> (32,7,7)
             -> Flatten (1568) -> head
   ```
   Head A is Dense(1568, 10) with 20,490 parameters. Head B is Dense(1568, 128) → ReLU → Dense(128, 10) with 206,922 parameters. Notably, 97% of head B's parameters are in its first dense layer. The conv layers have only 4,800.
8. **Training protocol.** Batch 64, lr 0.05, same seed for fair comparisons. After each epoch the validation loss is checked, and the weights from the best epoch are kept and restored at the end.
9. **Choosing a head.** At 5,000 per class, A reached 77.8% and B 77.6% validation accuracy. That is a tie inside the noise. B was chosen because it reaches its best in about 5 epochs against 18 for A. A 0.2-point gap is far smaller than the ±1 noise, so it said nothing about which head is better.
10. **First test.** The test set was used once: head B 78.8%, baseline 70.0%. After that the test set was considered spent.
11. **Error analysis.** A confusion matrix (rows = true class, columns = predicted) and a look at the actual drawings. The errors are concentrated in look-alike classes: firetruck ↔ ambulance and tractor ↔ bulldozer. Many are plain boxes on wheels with no ladder or cross, and some doodles are unreadable, so part of the error is probably irreducible.
12. **More convolution layers.** Two 3×3 convs before each pool (218k parameters) improved validation accuracy by about 1 to 2 points (79.7% and 79.4% on two seeds, against 77.6% and 78.2% for head B). The gain came mostly from train, tractor and bulldozer, not from the firetruck/ambulance pair as first expected.
13. **Augmentation.** Flip + shift raised head B to **82.5% validation**, the largest single gain (about +4 points), and the train–validation gap disappeared. It was still improving at epoch 20.
14. **Fresh test set.** Because the first test set had been used while making decisions, a new one was drawn (`data_tools/make_fresh_test.py`): 10,000 drawings, 1,000 per class, excluding every drawing whose pixels match anything in the old train, validation or test splits (overlap checked: 0). The final models were scored on it once.
15. **Packaging.** Tests, weights, figures, a README, a requirements file, and a GitHub repository, with the scripts grouped by purpose.

---

## Part 4. Results and lessons

| Model | Validation | Fresh test (95% margin) |
|---|---|---|
| Dense baseline | 69.3% | 69.6% ± 0.9 |
| CNN head A | 77.8% | not scored |
| CNN head B | 77.6% | 78.4% ± 0.8 |
| Stacked convs | 79.7% | not scored |
| **CNN head B + augmentation (final)** | 82.5% | **82.6% ± 0.7** |

**What we learned**
- The CNN architecture gives about +9 points over the dense baseline, and augmentation adds about +4 more.
- Augmentation beat extra layers. When a model overfits, more varied data helps more than more capacity.
- Validation and test numbers agreed within about a point, so choosing models and epochs on validation did not inflate the results.
- Single runs vary by roughly 0.3 to 0.6 points between seeds, so differences below about 1 point should not be trusted.
- Measure before optimising: profiling showed the slowdowns came from the machine (CPU throttling made some epochs 4 to 6 times slower), not from the code.

**Limitations and not done**
- A single final run, and no published benchmark for this 10-class subset, so the absolute number has no external reference.
- Only 5,000 of the 116,000 to 220,000 drawings per class were used.
- Not tried: stacked convs with augmentation (the run was stopped after 4 epochs), longer training of head B + augmentation, richer augmentation, dropout, weight decay, batch normalisation, Adam, early stopping, a GPU, and a PyTorch comparison.

To run everything yourself, see `README.md`.