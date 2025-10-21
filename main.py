import math
import random

import matplotlib.pyplot as plt
import numpy as np

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

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

from matplotlib.widgets import TextBox, Button

class MidpointDisplacement:
    def __init__(self, size=129, roughness=0.5, seed=None):
        self.size = size
        self.roughness = roughness
        self.height_map = np.zeros((size, size))
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_2d(self, steps=8):
        """Генерация 2D горного массива"""
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

    def generate_3d(self, steps=7):
        """Генерация 3D горного массива алгоритмом diamond-square"""
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
            for y in range(half_step, size - 1, step_size):
                for x in range(half_step, size - 1, step_size):
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


class InteractiveMidpointDisplacement:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 12))

        # Параметры по умолчанию
        self.params = {
            'roughness_3d': 0.7,
            'steps_3d': 6,
            'roughness_2d': 0.5,
            'steps_2d': 8,
            'seed': 42
        }

        self.setup_ui()
        self.update_plots()

    def setup_ui(self):
        """Создание интерфейса пользователя"""
        self.ax_3d = self.fig.add_axes([0.05, 0.55, 0.4, 0.4], projection='3d')
        self.ax_2d_steps = self.fig.add_axes([0.55, 0.55, 0.4, 0.4])
        self.ax_2d_roughness = self.fig.add_axes([0.05, 0.05, 0.4, 0.4])
        self.ax_single_line = self.fig.add_axes([0.55, 0.05, 0.4, 0.4])

        self.ax_roughness_3d = plt.axes([0.1, 0.48, 0.1, 0.03])
        self.ax_steps_3d = plt.axes([0.25, 0.48, 0.1, 0.03])
        self.ax_roughness_2d = plt.axes([0.4, 0.48, 0.1, 0.03])
        self.ax_steps_2d = plt.axes([0.55, 0.48, 0.1, 0.03])
        self.ax_seed = plt.axes([0.7, 0.48, 0.1, 0.03])
        self.ax_update = plt.axes([0.85, 0.48, 0.1, 0.03])

        self.text_roughness_3d = TextBox(self.ax_roughness_3d, 'Roughness 3D:',
                                         initial=str(self.params['roughness_3d']))
        self.text_steps_3d = TextBox(self.ax_steps_3d, 'Steps 3D:',
                                     initial=str(self.params['steps_3d']))
        self.text_roughness_2d = TextBox(self.ax_roughness_2d, 'Roughness 2D:',
                                         initial=str(self.params['roughness_2d']))
        self.text_steps_2d = TextBox(self.ax_steps_2d, 'Steps 2D:',
                                     initial=str(self.params['steps_2d']))
        self.text_seed = TextBox(self.ax_seed, 'Seed:',
                                 initial=str(self.params['seed']))

        self.button_update = Button(self.ax_update, 'ОБНОВИТЬ',
                                    color='lightblue', hovercolor='lightgreen')

        self.text_roughness_3d.on_submit(self.on_roughness_3d_change)
        self.text_steps_3d.on_submit(self.on_steps_3d_change)
        self.text_roughness_2d.on_submit(self.on_roughness_2d_change)
        self.text_steps_2d.on_submit(self.on_steps_2d_change)
        self.text_seed.on_submit(self.on_seed_change)
        self.button_update.on_clicked(self.on_update_clicked)

    def on_roughness_3d_change(self, text):
        try:
            self.params['roughness_3d'] = float(text)
            self.update_plots()
        except ValueError:
            pass

    def on_steps_3d_change(self, text):
        try:
            self.params['steps_3d'] = int(text)
            self.update_plots()
        except ValueError:
            pass

    def on_roughness_2d_change(self, text):
        try:
            self.params['roughness_2d'] = float(text)
            self.update_plots()
        except ValueError:
            pass

    def on_steps_2d_change(self, text):
        try:
            self.params['steps_2d'] = int(text)
            self.update_plots()
        except ValueError:
            pass

    def on_seed_change(self, text):
        try:
            self.params['seed'] = int(text)
            self.update_plots()
        except ValueError:
            pass

    def on_update_clicked(self, event):
        self.update_plots()

    def update_plots(self):
        """Обновление всех графиков"""
        self.ax_3d.clear()
        self.ax_2d_steps.clear()
        self.ax_2d_roughness.clear()
        self.ax_single_line.clear()

        # 1. 3D ландшафт
        self.ax_3d.set_title('3D Горный массив', fontsize=12, fontweight='bold')
        md_3d = MidpointDisplacement(roughness=self.params['roughness_3d'], seed=self.params['seed'])
        landscape_3d = md_3d.generate_3d(steps=self.params['steps_3d'])

        x = np.linspace(0, 1, landscape_3d.shape[0])
        y = np.linspace(0, 1, landscape_3d.shape[1])
        X, Y = np.meshgrid(x, y)

        surf = self.ax_3d.plot_surface(X, Y, landscape_3d, cmap='terrain', alpha=0.9, linewidth=0)
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')

        # 2. 2D последовательные шаги
        self.ax_2d_steps.set_title('2D: Последовательные шаги алгоритма', fontsize=12, fontweight='bold')
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        steps_list = [1, 2, 4, 8]

        for steps, color in zip(steps_list[:min(5, self.params['steps_2d'])], colors):
            md_2d = MidpointDisplacement(roughness=self.params['roughness_2d'], seed=self.params['seed'])
            heights = md_2d.generate_2d(steps=steps)
            x = np.linspace(0, 1, len(heights))
            self.ax_2d_steps.plot(x, heights, color=color, label=f'{steps} шагов', linewidth=1.0, alpha=0.8)

        self.ax_2d_steps.legend(fontsize=8)
        self.ax_2d_steps.grid(True, alpha=0.3)

        # 3. Различные параметры шероховатости
        self.ax_2d_roughness.set_title('Различная шероховатость', fontsize=12, fontweight='bold')
        roughness_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        for roughness, color in zip(roughness_values, colors):
            md_2d = MidpointDisplacement(roughness=roughness, seed=self.params['seed'])
            heights = md_2d.generate_2d(steps=self.params['steps_2d'])
            x = np.linspace(0, 1, len(heights))
            self.ax_2d_roughness.plot(x, heights, color=color, label=f'R={roughness}', linewidth=1.0, alpha=0.8)

        self.ax_2d_roughness.legend(fontsize=8)
        self.ax_2d_roughness.grid(True, alpha=0.3)

        # 4. Одна линия Midpoint Displacement
        self.ax_single_line.set_title('Midpoint Displacement', fontsize=12, fontweight='bold')
        md_single = MidpointDisplacement(roughness=self.params['roughness_2d'], seed=self.params['seed'])
        single_line = md_single.generate_2d(steps=self.params['steps_2d'])
        x_single = np.linspace(0, 1, len(single_line))

        self.ax_single_line.plot(x_single, single_line, 'b-', linewidth=1.5, alpha=0.8, label='Midpoint Displacement')
        self.ax_single_line.grid(True, alpha=0.3)
        self.ax_single_line.legend(fontsize=10)

        plt.draw()


def task2_midpoint_displacement():
    """Задание 2: Midpoint Displacement"""
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 2: ALGORITHM MIDPOINT DISPLACEMENT")
    print("=" * 60)

    app = InteractiveMidpointDisplacement()
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
    print("3. Все задания вместе")
    print("0. Выход")

    while True:
        try:
            choice = input("\nВаш выбор (0-3): ").strip()
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
                print("\nЗапуск всех заданий...")
                task1_l_systems()
                task2_midpoint_displacement()
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
    interactive_demo()
