import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

x_start = -5
x_end = 5
n_points = 100
coeff_a = 0.8  # Истинный коэффициент a
coeff_b = -1.5  # Истинный коэффициент b
coeff_c = 2.0  # Истинный коэффициент c
noise_level = 5  # Амплитуда шума

def create_dataset():
    x_vals = np.linspace(x_start, x_end, n_points)
    y_vals = coeff_a * x_vals**2 + coeff_b * x_vals + coeff_c + np.random.uniform(-noise_level, noise_level, size=len(x_vals))
    return x_vals, y_vals

x_data, y_data = create_dataset()

def calc_grad_a(x, y, a, b, c):
    return (2 / len(x)) * np.sum(x**2 * ((a * x**2 + b * x + c) - y))

def calc_grad_b(x, y, a, b, c):
    return (2 / len(x)) * np.sum(x * ((a * x**2 + b * x + c) - y))

def calc_grad_c(x, y, a, b, c):
    return (2 / len(x)) * np.sum((a * x**2 + b * x + c) - y)

def calculate_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def train_model(x, y, lr=0.001, n_iters=1000, init_a=0, init_b=0, init_c=0):
    a_curr, b_curr, c_curr = init_a, init_b, init_c
    log = {'a': [a_curr], 'b': [b_curr], 'c': [c_curr], 'error': [calculate_error(y, a_curr * x**2 + b_curr * x + c_curr)]}
    for _ in range(n_iters):
        grad_a = calc_grad_a(x, y, a_curr, b_curr, c_curr)
        grad_b = calc_grad_b(x, y, a_curr, b_curr, c_curr)
        grad_c = calc_grad_c(x, y, a_curr, b_curr, c_curr)
        a_curr -= lr * grad_a
        b_curr -= lr * grad_b
        c_curr -= lr * grad_c
        log['a'].append(a_curr)
        log['b'].append(b_curr)
        log['c'].append(c_curr)
        log['error'].append(calculate_error(y, a_curr * x**2 + b_curr * x + c_curr))
    return log

learning_rate = 0.001
iterations = 500
start_a, start_b, start_c = 0, 0, 0
training_log = train_model(x_data, y_data, lr=learning_rate, n_iters=iterations,
                         init_a=start_a, init_b=start_b, init_c=start_c)

fig, (ax_reg, ax_err) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)
data_plot = ax_reg.scatter(x_data, y_data, c='blue', label='Данные', alpha=0.6)
x_range = np.linspace(x_start, x_end, 100)
reg_line, = ax_reg.plot(x_range, training_log['a'][0] * x_range**2 + training_log['b'][0] * x_range + training_log['c'][0],
                       'r-', linewidth=2, label='Аппроксимация')
ax_reg.set_title(f'Квадратичная регрессия (Ошибка: {training_log["error"][0]:.2f})')
ax_reg.legend()
ax_reg.grid(True)
err_line, = ax_err.plot(training_log['error'], 'g-')
ax_err.set_title('Среднеквадратичная ошибка')
ax_err.grid(True)

slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
epoch_slider = Slider(slider_ax, 'Итерация', 0, iterations, valinit=0, valstep=1)

def update_plot(val):
    iter_num = int(epoch_slider.val)
    curr_a = training_log['a'][iter_num]
    curr_b = training_log['b'][iter_num]
    curr_c = training_log['c'][iter_num]
    reg_line.set_ydata(curr_a * x_range**2 + curr_b * x_range + curr_c)
    ax_reg.set_title(f'Квадратичная регрессия (Итерация: {iter_num}, Ошибка: {training_log["error"][iter_num]:.2f})')
    err_line.set_data(range(iter_num + 1), training_log['error'][:iter_num + 1])
    ax_err.set_xlim(0, max(iter_num, 10))
    ax_err.set_ylim(0, max(training_log['error'][:iter_num + 1]) * 1.1)
    fig.canvas.draw_idle()

epoch_slider.on_changed(update_plot)
plt.show()

final_a = training_log['a'][-1]
final_b = training_log['b'][-1]
final_c = training_log['c'][-1]
print(f"Истинные коэффициенты: a = {coeff_a}, b = {coeff_b}, c = {coeff_c}")
print(f"Полученные коэффициенты: a = {final_a:.4f}, b = {final_b:.4f}, c = {final_c:.4f}")
print(f"Финальная ошибка: {training_log['error'][-1]:.4f}")