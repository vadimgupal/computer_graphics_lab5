import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import random
import math
import json
from typing import List, Tuple, Dict, Any
import os

plt.rcParams['font.family'] = 'DejaVu Sans'


# =============================================================================
# ЗАДАНИЕ 1: L-СИСТЕМЫ (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# =============================================================================

class LSystem:
    def __init__(self):
        self.axiom = ""
        self.rules = {}
        self.angle = 0
        self.initial_angle = 0
        self.distance = 10
        self.randomness = 0.0

    def load_from_file(self, filename: str):
        """Загрузка L-системы из файла согласно формату задания"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            if not lines:
                raise ValueError("Файл пустой")

            # Первая строка: <атом> <угол поворота> <начальное направление>
            first_line = lines[0].split()
            if len(first_line) < 2:
                raise ValueError("Неверный формат первой строки")

            self.axiom = first_line[0]
            self.angle = float(first_line[1])
            self.initial_angle = float(first_line[2]) if len(first_line) > 2 else 0

            # Остальные строки: правила
            self.rules = {}
            for line in lines[1:]:
                if '->' in line:
                    key, value = line.split('->', 1)  # Разделяем только по первому '->'
                    self.rules[key.strip()] = value.strip()
                else:
                    print(f"Предупреждение: строка '{line}' не является правилом")

            print(f"Загружена L-система из {filename}:")
            print(f"  Аксиома: {self.axiom}")
            print(f"  Угол: {self.angle}")
            print(f"  Начальный угол: {self.initial_angle}")
            print(f"  Правила: {self.rules}")

        except Exception as e:
            print(f"Ошибка загрузки файла {filename}: {e}")
            raise

    def generate_string(self, iterations: int) -> str:
        """Генерация строки L-системы"""
        current = self.axiom
        for _ in range(iterations):
            next_string = ""
            for char in current:
                if char in self.rules:
                    # Добавляем случайность если есть параметр случайности
                    if self.randomness > 0 and random.random() < self.randomness:
                        # С вероятностью randomness оставляем исходный символ
                        if random.random() > 0.5:
                            next_string += char
                        else:
                            next_string += self.rules[char]
                    else:
                        next_string += self.rules[char]
                else:
                    next_string += char
            current = next_string
        return current

    def draw(self, iterations: int, ax, color='black', line_width=2):
        """Отрисовка L-системы"""
        string = self.generate_string(iterations)

        x, y = 0, 0
        angle = self.initial_angle
        stack = []  # Стек для ветвлений
        lines = []  # Список отрезков для отрисовки

        current_width = line_width
        current_color = color

        i = 0
        while i < len(string):
            char = string[i]

            if char == 'F' or char == 'G' or char == 'A' or char == 'B':
                # Движение вперед
                new_x = x + self.distance * math.cos(math.radians(angle))
                new_y = y + self.distance * math.sin(math.radians(angle))
                lines.append(((x, y), (new_x, new_y), current_width, current_color))
                x, y = new_x, new_y

            elif char == 'f':
                # Движение вперед без отрисовки
                new_x = x + self.distance * math.cos(math.radians(angle))
                new_y = y + self.distance * math.sin(math.radians(angle))
                x, y = new_x, new_y

            elif char == '+':
                # Поворот налево
                angle += self.angle
                if self.randomness > 0:
                    angle += random.uniform(-self.angle * self.randomness, self.angle * self.randomness)

            elif char == '-':
                # Поворот направо
                angle -= self.angle
                if self.randomness > 0:
                    angle += random.uniform(-self.angle * self.randomness, self.angle * self.randomness)

            elif char == '[':
                # Сохраняем текущее состояние
                stack.append((x, y, angle, current_width, current_color))
                # Уменьшаем толщину и меняем цвет для ветвей
                current_width *= 0.7
                if current_color == 'brown' or current_color == '#8B4513':
                    # Переход от коричневого к зеленому
                    current_color = 'green'

            elif char == ']':
                # Восстанавливаем состояние
                if stack:
                    x, y, angle, current_width, current_color = stack.pop()

            i += 1

        # Отрисовка всех линий
        for (x1, y1), (x2, y2), width, color_val in lines:
            ax.plot([x1, x2], [y1, y2], color=color_val, linewidth=width)

        # Масштабирование
        ax.autoscale_view()
        ax.set_aspect('equal')


def create_lsystem_files():
    """Создание тестовых файлов L-систем согласно заданию"""
    files_content = {
        # 1. Кривая Коха
        'koch.txt': """F 60 0
F->F+F--F+F""",

        # 2. Ковер Серпинского
        'sierpinski.txt': """F 120 0
F->F-G+F+G-F
G->GG""",

        # 3. Кривая дракона
        'dragon.txt': """FX 90 0
X->X+YF+
Y->-FX-Y""",

        # 4. Фрактальное дерево (простое)
        'tree_simple.txt': """F 25 90
F->FF+[+F-F-F]-[-F+F+F]""",

        # 5. Фрактальное дерево (сложное, с ветвлением)
        'tree_advanced.txt': """A 22 90
A->F[+A][-A]
F->FF""",

        # 6. Куст
        'bush.txt': """F 22 90
F->FF-[-F+F+F]+[+F-F-F]""",

        # 7. Квадратный остров Коха
        'koch_island.txt': """F+F+F+F 90 0
F->F+F-F-FF+F+F-F"""
    }

    created_files = []
    for filename, content in files_content.items():
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files.append(filename)
            print(f"Создан файл: {filename}")
        except Exception as e:
            print(f"Ошибка создания файла {filename}: {e}")

    return created_files


def task1_l_systems():
    """Задание 1: L-системы с загрузкой из файлов"""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 1: L-СИСТЕМЫ (работа с файлами)")
    print("=" * 60)

    # Создаем тестовые файлы
    lsystem_files = create_lsystem_files()

    if not lsystem_files:
        print("Не удалось создать файлы L-систем!")
        return

    # Создаем график
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ЗАДАНИЕ 1: L-СИСТЕМЫ - Фракталы из файлов', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    # Загружаем и отрисовываем каждую L-систему из файла
    for i, filename in enumerate(lsystem_files[:6]):  # Показываем первые 6
        if i >= len(axes):
            break

        ax = axes[i]
        try:
            # Загружаем L-систему из файла
            lsystem = LSystem()
            lsystem.load_from_file(filename)
            lsystem.distance = 15 - i * 2  # Настраиваем расстояние для лучшего отображения

            # Для дерева добавляем особенности
            if 'tree' in filename:
                lsystem.randomness = 0.2
                color = 'brown'
                line_width = 2
            else:
                color = ['blue', 'red', 'green', 'purple', 'orange', 'brown'][i]
                line_width = 1

            # Отрисовываем
            iterations = 4 if 'tree' in filename or 'bush' in filename else 5
            lsystem.draw(iterations, ax, color, line_width)

            ax.set_title(f'{filename}\n{lsystem.axiom} -> ...')
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(0.5, 0.5, f'Ошибка:\n{str(e)}',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Ошибка: {filename}')

    # Убираем лишние subplots
    for i in range(len(lsystem_files), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Демонстрация работы с деревом (задание 1.б)
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 1.б: Фрактальное дерево с улучшениями")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Дерево из файла
    try:
        tree_system = LSystem()
        tree_system.load_from_file('tree_advanced.txt')
        tree_system.distance = 8
        tree_system.randomness = 0.3  # Случайность углов

        ax1.set_title('Дерево из файла tree_advanced.txt')
        tree_system.draw(5, ax1, 'brown', 3)
        ax1.grid(True, alpha=0.3)

    except Exception as e:
        ax1.text(0.5, 0.5, f'Ошибка загрузки:\n{str(e)}',
                 transform=ax1.transAxes, ha='center', va='center')

    # Дерево с дополнительными улучшениями
    try:
        custom_tree = LSystem()
        custom_tree.load_from_file('tree_advanced.txt')
        custom_tree.distance = 6
        custom_tree.randomness = 0.4  # Больше случайности
        custom_tree.angle = 30  # Изменяем угол

        ax2.set_title('Дерево с увеличенной случайностью')
        custom_tree.draw(5, ax2, '#8B4513', 4)  # Темно-коричневый
        ax2.grid(True, alpha=0.3)

    except Exception as e:
        ax2.text(0.5, 0.5, f'Ошибка:\n{str(e)}',
                 transform=ax2.transAxes, ha='center', va='center')

    plt.tight_layout()
    plt.show()


# =============================================================================
# ЗАДАНИЕ 2: ALGORITHM MIDPOINT DISPLACEMENT
# =============================================================================

class MidpointDisplacement:
    def __init__(self, size=129, roughness=0.5, seed=None):
        self.size = size
        self.roughness = roughness
        self.height_map = np.zeros((size, size))
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_1d(self, steps=8):
        """Генерация 1D горного массива"""
        size = 2 ** steps + 1
        heights = np.zeros(size)

        # Начальные значения
        heights[0] = random.uniform(-1, 1)
        heights[-1] = random.uniform(-1, 1)

        step_size = size - 1
        scale = 1.0

        while step_size > 1:
            half_step = step_size // 2

            # Diamond step (для 1D это просто midpoint)
            for i in range(half_step, size - 1, step_size):
                heights[i] = (heights[i - half_step] + heights[i + half_step]) / 2
                heights[i] += random.uniform(-scale, scale) * self.roughness

            step_size = half_step
            scale *= 0.5

        return heights

    def generate_2d(self, steps=7):
        """Генерация 2D горного массива алгоритмом diamond-square"""
        size = 2 ** steps + 1
        self.height_map = np.zeros((size, size))

        # Инициализация углов
        self.height_map[0, 0] = random.uniform(-1, 1)
        self.height_map[0, -1] = random.uniform(-1, 1)
        self.height_map[-1, 0] = random.uniform(-1, 1)
        self.height_map[-1, -1] = random.uniform(-1, 1)

        step_size = size - 1
        scale = 1.0

        while step_size > 1:
            half_step = step_size // 2

            # Diamond step
            for y in range(half_step, size, step_size):
                for x in range(half_step, size, step_size):
                    avg = (self.height_map[y - half_step, x - half_step] +
                           self.height_map[y - half_step, x + half_step] +
                           self.height_map[y + half_step, x - half_step] +
                           self.height_map[y + half_step, x + half_step]) / 4
                    self.height_map[y, x] = avg + random.uniform(-scale, scale) * self.roughness

            # Square step
            for y in range(0, size, half_step):
                for x in range((y + half_step) % step_size, size, step_size):
                    total = 0
                    count = 0

                    # Сверху
                    if y - half_step >= 0:
                        total += self.height_map[y - half_step, x]
                        count += 1
                    # Снизу
                    if y + half_step < size:
                        total += self.height_map[y + half_step, x]
                        count += 1
                    # Слева
                    if x - half_step >= 0:
                        total += self.height_map[y, x - half_step]
                        count += 1
                    # Справа
                    if x + half_step < size:
                        total += self.height_map[y, x + half_step]
                        count += 1

                    if count > 0:
                        self.height_map[y, x] = total / count + random.uniform(-scale, scale) * self.roughness

            step_size = half_step
            scale *= 0.5

        return self.height_map


def task2_midpoint_displacement():
    """Задание 2: Алгоритм Midpoint Displacement"""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 2: ALGORITHM MIDPOINT DISPLACEMENT")
    print("=" * 60)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('ЗАДАНИЕ 2: ALGORITHM MIDPOINT DISPLACEMENT', fontsize=16, fontweight='bold')

    # 2D ландшафт
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_title('2D: 3D Горный массив')

    md_2d = MidpointDisplacement(size=65, roughness=0.7, seed=42)
    landscape_2d = md_2d.generate_2d(steps=6)

    x = np.linspace(0, 1, landscape_2d.shape[0])
    y = np.linspace(0, 1, landscape_2d.shape[1])
    X, Y = np.meshgrid(x, y)

    surf = ax1.plot_surface(X, Y, landscape_2d, cmap='terrain', alpha=0.9)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Высота')

    # 1D последовательные шаги
    ax2 = fig.add_subplot(222)
    ax2.set_title('1D: Последовательные шаги алгоритма')

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for steps, color in zip([1, 2, 4, 6, 8], colors):
        md_1d = MidpointDisplacement(roughness=0.5, seed=42)
        heights = md_1d.generate_1d(steps=steps)
        x = np.linspace(0, 1, len(heights))
        ax2.plot(x, heights, color=color, label=f'{steps} шагов', linewidth=2)

    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Позиция')
    ax2.set_ylabel('Высота')

    # Различные параметры шероховатости
    ax3 = fig.add_subplot(223)
    ax3.set_title('Различная шероховатость')

    roughness_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for roughness, color in zip(roughness_values, colors):
        md_1d = MidpointDisplacement(roughness=roughness, seed=42)
        heights = md_1d.generate_1d(steps=8)
        x = np.linspace(0, 1, len(heights))
        ax3.plot(x, heights, color=color, label=f'R={roughness}', linewidth=2)

    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Позиция')
    ax3.set_ylabel('Высота')

    # 2D вид сверху
    ax4 = fig.add_subplot(224)
    ax4.set_title('2D: Вид сверху (heatmap)')

    im = ax4.imshow(landscape_2d, cmap='terrain', extent=[0, 1, 0, 1])
    plt.colorbar(im, ax=ax4, label='Высота')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')

    plt.tight_layout()
    plt.show()


# =============================================================================
# ЗАДАНИЕ 3: КУБИЧЕСКИЕ СПЛАЙНЫ БЕЗЬЕ
# =============================================================================

class BezierSpline:
    def __init__(self):
        self.control_points = []
        self.curves = []  # Список кубических кривых Безье

    def add_control_point(self, x, y):
        """Добавление контрольной точки"""
        self.control_points.append((x, y))
        self._update_curves()

    def remove_control_point(self, index):
        """Удаление контрольной точки"""
        if 0 <= index < len(self.control_points):
            self.control_points.pop(index)
            self._update_curves()

    def move_control_point(self, index, x, y):
        """Перемещение контрольной точки"""
        if 0 <= index < len(self.control_points):
            self.control_points[index] = (x, y)
            self._update_curves()

    def _update_curves(self):
        """Обновление составной кривой Безье"""
        self.curves = []
        n = len(self.control_points)

        if n < 4:
            return

        # Создаем составную кривую Безье
        # Каждые 4 точки определяют кубическую кривую
        for i in range(0, n - 3, 3):
            p0 = np.array(self.control_points[i])
            p1 = np.array(self.control_points[i + 1])
            p2 = np.array(self.control_points[i + 2])
            p3 = np.array(self.control_points[i + 3])
            self.curves.append((p0, p1, p2, p3))

    def bezier_curve(self, p0, p1, p2, p3, num_points=100):
        """Вычисление точек кубической кривой Безье"""
        t = np.linspace(0, 1, num_points)
        curve_x = []
        curve_y = []

        for t_val in t:
            # Кубическая формула Безье
            point = (1 - t_val) ** 3 * p0 + 3 * (1 - t_val) ** 2 * t_val * p1 + 3 * (
                        1 - t_val) * t_val ** 2 * p2 + t_val ** 3 * p3
            curve_x.append(point[0])
            curve_y.append(point[1])

        return curve_x, curve_y

    def draw(self, ax):
        """Отрисовка сплайна и контрольных точек"""
        # Отрисовка контрольных точек
        if self.control_points:
            points_x, points_y = zip(*self.control_points)
            ax.scatter(points_x, points_y, color='red', s=50, zorder=5, label='Контрольные точки')

            # Соединяем контрольные точки
            ax.plot(points_x, points_y, 'r--', alpha=0.5, linewidth=1, label='Контрольный полигон')

        # Отрисовка кривых Безье
        for i, (p0, p1, p2, p3) in enumerate(self.curves):
            curve_x, curve_y = self.bezier_curve(p0, p1, p2, p3)
            ax.plot(curve_x, curve_y, 'b-', linewidth=2, label='Кривая Безье' if i == 0 else "")

            # Показываем касательные
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'g--', alpha=0.7, linewidth=1)
            ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 'g--', alpha=0.7, linewidth=1)


def task3_bezier_splines():
    """Задание 3: Кубические сплайны Безье"""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 3: КУБИЧЕСКИЕ СПЛАЙНЫ БЕЗЬЕ")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('ЗАДАНИЕ 3: КУБИЧЕСКИЕ СПЛАЙНЫ БЕЗЬЕ', fontsize=16, fontweight='bold')

    spline = BezierSpline()

    # Добавляем начальные точки для демонстрации
    demo_points = [
        (1, 1), (2, 3), (4, 2), (5, 4),
        (6, 1), (8, 3), (9, 1), (11, 2)
    ]

    for point in demo_points:
        spline.add_control_point(*point)

    spline.draw(ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Составная кубическая кривая Безье\n(Каждые 4 точки определяют сегмент кривой)')
    ax.set_aspect('equal')

    # Добавляем поясняющий текст
    text_info = """
    Составная кубическая кривая Безье:
    • Красные точки - контрольные точки
    • Красный пунктир - контрольный полигон  
    • Синие линии - кривые Безье
    • Зеленый пунктир - касательные векторы

    Структура: P0, P1, P2, P3, P4, P5, P6, P7...
    Кривые: [P0-P3], [P3-P6], ...
    """
    ax.text(0.02, 0.98, text_info, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

    plt.tight_layout()
    plt.show()


# =============================================================================
# ИНТЕРАКТИВНАЯ ДЕМОНСТРАЦИЯ
# =============================================================================

def interactive_demo():
    """Интерактивная демонстрация всех трех заданий"""
    print("=" * 60)
    print("КОМПЬЮТЕРНАЯ ГРАФИКА: ФРАКТАЛЫ И СПЛАЙНЫ")
    print("=" * 60)
    print("\nВыберите задание для демонстрации:")
    print("1. L-системы (фрактальные узоры)")
    print("2. Midpoint Displacement (горные массивы)")
    print("3. Кубические сплайны Безье")
    print("4. Все задания вместе")
    print("0. Выход")

    while True:
        try:
            choice = input("\nВаш выбор (0-4): ").strip()
            if choice == '0':
                print("Выход из программы.")
                break
            elif choice == '1':
                print("\nЗапуск задания 1: L-системы...")
                task1_l_systems()
            elif choice == '2':
                print("\nЗапуск задания 2: Midpoint Displacement...")
                task2_midpoint_displacement()
            elif choice == '3':
                print("\nЗапуск задания 3: Кубические сплайны Безье...")
                task3_bezier_splines()
            elif choice == '4':
                print("\nЗапуск всех заданий...")
                task1_l_systems()
                task2_midpoint_displacement()
                task3_bezier_splines()
            else:
                print("Неверный выбор. Попробуйте снова.")
        except KeyboardInterrupt:
            print("\nПрограмма завершена.")
            break
        except Exception as e:
            print(f"Произошла ошибка: {e}")


# =============================================================================
# ОСНОВНАЯ ПРОГРАММА
# =============================================================================

if __name__ == "__main__":
    # Создаем демонстрационные файлы L-систем
    demo_files = {
        'koch.txt': """F 60 0
F->F+F--F+F""",

        'tree.txt': """A 25 90
A->F[+A][-A]
F->FF""",

        'dragon.txt': """FX 90 0
X->X+YF+
Y->-FX-Y"""
    }

    # Сохраняем демо файлы
    for filename, content in demo_files.items():
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        except:
            pass

    print("ДЕМОНСТРАЦИОННАЯ ПРОГРАММА ПО КОМПЬЮТЕРНОЙ ГРАФИКЕ")
    print("Реализованы все три задания:")
    print("1. L-системы для фрактальных узоров")
    print("2. Алгоритм Midpoint Displacement для горных массивов")
    print("3. Кубические сплайны Безье для гладких кривых")
    print("\n" + "=" * 60)

    # Запускаем интерактивную демонстрацию
    interactive_demo()
