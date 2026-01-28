"""
Практическая работа 1. Вариант 9.
Нейронная сеть для вычисления линейной функции y = 3x + 5
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import os

# Убираем лишние предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Создание обучающих данных
np.random.seed(42)
x_train = np.random.uniform(-10, 10, 1000).reshape(-1, 1)  # Важно: reshape для 2D массива
y_train = 3 * x_train + 5

# 2. Создание модели нейронной сети (новый синтаксис)
inputs = Input(shape=(1,))
outputs = Dense(1, activation='linear')(inputs)
model = Model(inputs=inputs, outputs=outputs)

# 3. Компиляция модели
model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.Adam(learning_rate=0.01)
)

# 4. Обучение модели
print("Начало обучения...")
history = model.fit(
    x_train, 
    y_train,
    epochs=100,
    batch_size=32,
    verbose=0
)

# 5. Проверка работы обученной сети
print("\nТестирование обученной сети:")
test_values = np.array([-5, 0, 2, 10, 15]).reshape(-1, 1)  # Преобразуем в 2D

print(f"{'x':>5} | {'Истинное y':>12} | {'Предсказанное y':>15} | {'Ошибка':>8}")
print("-" * 55)

for i, x in enumerate(test_values):
    true_y = 3 * x[0] + 5
    # Правильный вызов predict для одного значения
    predicted_y = model.predict(x.reshape(1, -1), verbose=0)[0][0]
    error = abs(true_y - predicted_y)
    print(f"{x[0]:5.1f} | {true_y:12.2f} | {predicted_y:15.2f} | {error:8.4f}")

# 6. Вывод весов сети
weights = model.get_weights()
if len(weights) >= 2:
    print(f"\nОбученные веса сети:")
    print(f"Коэффициент (должен быть близок к 3): {weights[0][0][0]:.6f}")
    print(f"Смещение (должен быть близок к 5): {weights[1][0]:.6f}")
else:
    print("\nНе удалось получить веса сети")

# 7. Визуализация процесса обучения
plt.figure(figsize=(12, 4))

# График функции потерь
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Функция потерь в процессе обучения')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.grid(True)

# График сравнения истинных и предсказанных значений
plt.subplot(1, 2, 2)
x_test_vis = np.linspace(-10, 10, 50).reshape(-1, 1)
y_true_vis = 3 * x_test_vis + 5
y_pred_vis = model.predict(x_test_vis, verbose=0)

plt.scatter(x_train[:100], y_train[:100], alpha=0.3, label='Обучающие данные')
plt.plot(x_test_vis, y_true_vis, 'r-', linewidth=2, label='Истинная функция')
plt.plot(x_test_vis, y_pred_vis, 'g--', linewidth=2, label='Предсказания НС')
plt.title('Сравнение истинной функции и предсказаний сети')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 8. Проверка на новых данных
print("\nПроверка на произвольных значениях:")
random_x = np.random.uniform(-20, 20, 5)
for x in random_x:
    true = 3 * x + 5
    pred = model.predict(np.array([[x]]), verbose=0)[0][0]
    print(f"x = {x:6.2f}: Истинное = {true:7.2f}, Предсказание = {pred:7.2f}, "
          f"Ошибка = {abs(true - pred):.4f}")

# 9. Тест на точность
print("\n" + "="*50)
print("Итоговая проверка точности:")
test_x = np.array([-8, -3, 1, 4, 7, 12, 18]).reshape(-1, 1)
predictions = model.predict(test_x, verbose=0)
print(f"{'x':>5} | {'Ожидается':>10} | {'Получено':>10} | {'Разница':>8}")
print("-" * 45)
for i in range(len(test_x)):
    x_val = test_x[i][0]
    expected = 3 * x_val + 5
    predicted = predictions[i][0]
    diff = abs(expected - predicted)
    print(f"{x_val:5.1f} | {expected:10.2f} | {predicted:10.2f} | {diff:8.4f}")

print("="*50)
print("Обучение завершено успешно!")