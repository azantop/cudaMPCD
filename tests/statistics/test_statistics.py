import numpy as np
import scipy.stats as stats
from _test_statistics import generate_gaussian_samples, generate_uniform_samples


def test_uniform_distribution():
    """Test uniform random number quality"""
    samples = generate_uniform_samples(100000, 12345, 67890)
    samples = np.array(samples)

    # Basic statistical tests
    assert 0.49 < np.mean(samples) < 0.51, "Mean should be ~0.5"
    assert 0.08 < np.var(samples) < 0.09, "Variance should be ~1/sqrt(12) ≈ 0.0833"

    # Kolmogorov-Smirnov test against uniform distribution
    ks_stat, p_value = stats.kstest(samples, "uniform")
    assert p_value > 0.01, f"KS test failed: p={p_value}"

    # Chi-square test for uniformity
    hist, _ = np.histogram(samples, bins=50, range=(0, 1))
    expected = len(samples) / 50
    chi2_stat, chi2_p = stats.chisquare(hist, expected)
    assert chi2_p > 0.01, f"Chi-square test failed: p={chi2_p}"

    print(f"Uniform tests passed: KS p={p_value:.3f}, χ² p={chi2_p:.3f}")


def test_gaussian_distribution():
    """Test Gaussian random number quality"""
    samples = generate_gaussian_samples(100000, 12345, 67890)
    samples = np.array(samples)

    # Statistical tests
    assert -0.01 < np.mean(samples) < 0.01, f"Mean should be ~0, got {np.mean(samples)}"
    assert 0.99 < np.std(samples) < 1.01, f"Std should be ~1, got {np.std(samples)}"

    # Shapiro-Wilk normality test (on subset due to sample size limits)
    subset = np.random.default_rng().choice(samples, 5000, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(subset)
    assert shapiro_p > 0.01, f"Shapiro-Wilk test failed: p={shapiro_p}"

    # Anderson-Darling test for normality
    ad_stat, ad_crit, ad_sig = stats.anderson(samples, "norm")
    assert ad_stat < ad_crit[2], (
        f"Anderson-Darling test failed: {ad_stat} > {ad_crit[2]}"
    )

    print(f"Gaussian tests passed: Shapiro p={shapiro_p:.3f}, AD stat={ad_stat:.3f}")


if __name__ == "__main__":
    test_uniform_distribution()
    test_gaussian_distribution()
