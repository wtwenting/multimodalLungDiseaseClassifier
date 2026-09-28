"""Neural network architectures for cough classification."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_baseline_2d_cnn(
    input_shape: tuple[int, int, int] = (64, 688, 1),
    num_classes: int = 2,
) -> keras.Model:
    """Build a simple baseline 2D CNN for Log-Mel spectrograms."""

    model = keras.Sequential(
        [
            # Input:
            # 64 Mel bands x 688 time frames x 1 channel
            keras.Input(
                shape=input_shape,
                name="log_mel_input",
            ),

            # -------------------------------------------------
            # Convolution block 1
            # -------------------------------------------------
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv1",
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool1",
            ),

            # -------------------------------------------------
            # Convolution block 2
            # -------------------------------------------------
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv2",
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool2",
            ),

            # -------------------------------------------------
            # Convolution block 3
            # -------------------------------------------------
            layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv3",
            ),

            # -------------------------------------------------
            # Convert feature maps into one feature vector
            # -------------------------------------------------
            layers.GlobalAveragePooling2D(
                name="global_average_pool",
            ),

            # -------------------------------------------------
            # Classification output
            # -------------------------------------------------
            layers.Dense(
                num_classes,
                activation="softmax",
                name="output",
            ),
        ],
        name="coughvid_baseline_2d_cnn",
    )

    return model

def test_baseline_2d_cnn() -> None:
    """Build and inspect the baseline CNN."""

    model = build_baseline_2d_cnn()

    print("\n" + "=" * 60)
    print("COUGHVID BASELINE 2D CNN")
    print("=" * 60)

    model.summary()

    print("\nInput shape:")
    print(model.input_shape)

    print("\nOutput shape:")
    print(model.output_shape)

    print("\nTrainable parameters:")
    print(
        f"{model.count_params():,}"
    )