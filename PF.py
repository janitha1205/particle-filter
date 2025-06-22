import numpy as np
import matplotlib.pyplot as plt
def predict(input, num_p, noice_p):

    particle = []
    for i in range(num_p):
        particle.append(input + np.random.rand() * noice_p)
    return particle


def pdf(x, mu, sigma):
    pdf_value = []
    for i in range(len(x)):
        PI = np.pi

        if sigma > 0:

            coefficient = 1.0 / (sigma * np.sqrt(2 * PI))

            exponent = -0.5 * ((x[i] - mu) / sigma) ** 2

            pdf_value.append(coefficient * np.exp(exponent))

    return pdf_value


def update_weights(particles, z, measurement_noise, weights_old):
    likelihood = pdf(particles, z, measurement_noise)
    weights = []

    for i in range(len(weights_old)):

        weights.append(weights_old[i] * likelihood[i])
    total_weight = np.sum(weights)
    weight2 = []
    if total_weight > 1e-15:
        for i in range(len(weights_old)):
            weight2.append(weights[i] / total_weight)

    else:
        # Handle case where all weights are zero
        weight2 = np.ones(len(weights)) / len(weights)
    return weight2


def resample(particles, weights):
    n_particles = len(particles)
    # Create a new set of particles by drawing from the old set,
    # where the probability of being chosen is proportional to its weight.

    new_particles = []
    for i in range(n_particles):
        indices = int(np.random.uniform(n_particles))
        new_particles.append(particles[indices])
    # After resampling, all particles are important again
    weights = np.ones(n_particles) / n_particles
    return new_particles, weights


def estimate(particles, weights):
    n_p = len(particles)
    n_w = len(weights)
    sum_p = 0
    sum_w = 0
    for i in range(n_p):
        sum_p += particles[i] * weights[i]
    for j in range(n_w):
        sum_w += weights[j]
    return (sum_p) / (sum_w)


def run_simulation(x, n_parti, weights):
    p_noice = 0.1
    m_noice = 0.4
    x_true = x + np.random.randn() * p_noice
    measurement = x_true + np.random.randn() * m_noice
    paticles = predict(x_true, n_parti, p_noice)
    weight = update_weights(paticles, measurement, m_noice, weights)
    paticles2, weight2 = resample(paticles, weight)
    estimate_n = estimate(paticles2, weight2)
    return x_true, measurement, estimate_n, weight2


def plot_p(y_axis, x1, x2, x3):
    plt.plot(y_axis, x1, y_axis, x2, y_axis, x3)
    plt.show()


def main():
    u = 1
    y = 0
    y_a = []
    x_true = []
    measurement = []
    estimate_n = []
    n_parti = 50
    weights = np.ones(n_parti) / n_parti
    steps = 100
    for step in range(steps):
        x = 1.5 * y + u

        x_true2, measurement2, estimate_n2, weights2 = run_simulation(
            x, n_parti, weights
        )
        x_true.append(x_true2)
        measurement.append(measurement2)
        estimate_n.append(estimate_n2)
        weights = []
        weights = weights2
        y_a.append(y)
        y += 0.1
    plot_p(y_a, x_true, measurement, estimate_n)


if __name__ == "__main__":
    main()
