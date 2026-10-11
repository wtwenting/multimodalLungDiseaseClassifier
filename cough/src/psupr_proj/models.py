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

def build_baseline_1d_cnn(
    input_shape: tuple[int, int] = (64000, 1),
    num_classes: int = 2,
) -> keras.Model:
    """Build a baseline 1D CNN that learns directly from raw waveforms."""

    return keras.Sequential(
        [
            keras.Input(shape=input_shape, name="waveform_input"),
            layers.Conv1D(32, 9, strides=2, padding="same", activation="relu", name="conv1"),
            layers.MaxPooling1D(pool_size=4, name="pool1"),
            layers.Conv1D(64, 7, strides=2, padding="same", activation="relu", name="conv2"),
            layers.MaxPooling1D(pool_size=4, name="pool2"),
            layers.Conv1D(128, 5, strides=2, padding="same", activation="relu", name="conv3"),
            layers.GlobalAveragePooling1D(name="global_average_pool"),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="coughvid_baseline_1d_cnn",
    )


def build_baseline_lstm(
    input_shape: tuple[int, int] = (251, 13),
    num_classes: int = 2,
) -> keras.Model:
    """Build Model 3: MFCC sequence -> LSTM -> binary softmax output."""

    return keras.Sequential(
        [
            keras.Input(shape=input_shape, name="mfcc_sequence_input"),
            layers.LSTM(64, name="lstm"),
            layers.Dense(64, activation="relu", name="dense"),
            layers.Dropout(0.3, name="dropout"),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="coughvid_baseline_lstm",
    )



def test_coswara_2d_cnn_forward_pass(
    batch_size: int = 8,
    input_shape: tuple[int, int, int] = (64, 626, 1),
    num_classes: int = 2,
) -> None:
    """Smoke-test the existing 2D CNN architecture with Coswara input.

    This performs a forward pass only. No model training or weight update occurs.
    """

    model = build_baseline_2d_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
    )

    # Synthetic input is sufficient for an architecture/shape smoke test.
    x = tf.zeros(
        (batch_size, *input_shape),
        dtype=tf.float32,
    )

    predictions = model(
        x,
        training=False,
    )

    expected_output_shape = (
        batch_size,
        num_classes,
    )

    print("\n" + "=" * 60)
    print("COSWARA 2D CNN FORWARD-PASS TEST")
    print("=" * 60)

    print(f"\nInput shape:       {tuple(x.shape)}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Output shape:      {tuple(predictions.shape)}")
    print(f"Output dtype:      {predictions.dtype}")
    print(f"Parameters:        {model.count_params():,}")

    if tuple(predictions.shape) != expected_output_shape:
        raise ValueError(
            "Unexpected Coswara CNN output shape: "
            f"{tuple(predictions.shape)}; "
            f"expected {expected_output_shape}"
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(predictions)
        ).numpy()
    ):
        raise ValueError(
            "Coswara CNN predictions contain NaN/Inf."
        )

    row_sums = tf.reduce_sum(
        predictions,
        axis=1,
    )

    if not bool(
        tf.reduce_all(
            tf.abs(row_sums - 1.0) < 1e-5
        ).numpy()
    ):
        raise ValueError(
            "Coswara softmax predictions do not sum to 1."
        )

    print(
        "\nCOSWARA 2D CNN FORWARD-PASS TEST PASSED."
    )
    print("No training was performed.")
