# Feature List

Here we list the complete set of 51 features extracted from the light
curves and used as inputs for the RF and XGBoost models.

## Basic Statistics

-   **`baseline`**: The median flux level of the light curve, serving as
    a robust baseline reference.

-   **`flux_min`**: The minimum normalized flux value observed.

-   **`flux_max`**: The maximum normalized flux value observed.

-   **`flux_mean`**: The arithmetic mean of all flux measurements.

-   **`flux_std`**: The standard deviation of the flux, quantifying
    overall variability.

-   **`flux_range`**: Total amplitude (maximum - minimum flux).

-   **`flux_skew`**: Skewness of the flux distribution (measure of
    asymmetry).

-   **`flux_kurtosis`**: Kurtosis of the flux distribution (measure of
    outliers).

## Morphological Features

-   **`primary_depth`**: Depth of the primary (deeper) eclipse relative
    to baseline.

-   **`secondary_depth`**: Depth of the secondary (shallower) eclipse
    relative to baseline.

-   **`primary_width`**: Fractional width of the primary eclipse at
    half-maximum depth.

-   **`secondary_width`**: Fractional width of the secondary eclipse at
    half-maximum depth.

-   **`oconnell_effect`**: Flux difference between the two maxima (Max
    I - Max II), serving as a proxy for light curve asymmetries caused
    by starspots or mass transfer activity.

-   **`ooe_range`**: Range (max - min) of flux measured during
    Out-of-Eclipse phases, quantifying the non-eclipse variability
    driven by ellipsoidal distortion and reflection effects.

-   **`ooe_std`**: Standard deviation of flux during Out-of-Eclipse
    phases.

## Fourier Descriptors

-   **`fourier_amp_{1..10}`**: Amplitudes of the first 10 harmonics from
    the FFT.

-   **`fourier_phase_{1..3}`**: Phase angles of the first 3 harmonics.

-   **`fourier_ratio_{2..4}`**: Amplitude ratios of higher harmonics
    relative to the first ($A_k/A_1$).

-   **`fourier_total_power`**: Sum of the squared amplitudes of all
    harmonics (total signal energy).

## Phase-Bin Features

-   **`phase_bin_{1..10}_mean`**: Mean flux within 10 equidistant phase
    bins (low-res profile).

-   **`phase_bin_diff_{1..9}`**: Difference in mean flux between
    consecutive phase bins (local gradients).
