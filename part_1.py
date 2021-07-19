import numpy as np
import copy
import scipy.linalg

def part1_1(p, k, i, j):
    return np.linalg.matrix_power(p, k)[i, j]

def part1_2(p, k, a):
    res = np.linalg.matrix_power(p, k)
    return np.linalg.multi_dot([a, res])

def for_part1_3(p, k):
    p2 = p
    for sample in range(2, k+1):
        p3 = np.zeros(p.shape)
        for ii in range(p.shape[0]):
            for jj in range(p.shape[1]):
                rez = 0
                for l in range(p.shape[0]):
                    if l != jj:
                        rez += p[ii, l] * p2[l, jj]
                p3[ii, jj] = rez
        p2 = p3
    return p2

def part1_3(p, k, i, j):
    rez = for_part1_3(p, k)
    return rez[i, j]

def for_part_4(p_c, p):
    p_c2 = p_c
    p_c3 = copy.copy(p_c)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            rez = 0
            for l in range(p.shape[0]):
                if l != j:
                    rez += p[i, l] * p_c2[l, j]
            p_c3[i, j] = rez
    p_c2 = p_c3
    return p_c2


def part1_4(p, k, i, j):
    rez = []
    p4 = copy.copy(for_part1_3(p, 1))
    rez.append(p4[i, j])
    for ii in range(2, k+1):
        p_new = for_part_4(p4, p)
        rez.append(p_new[i, j])
        p4 = p_new
    return sum(rez)

def part1_5(p, i, j):
    rez = []
    p5 = copy.copy(for_part1_3(p, 1))
    rez.append(p5[i, j])
    for key in range(2, 10000):
        p5_new = for_part_4(p5, p)
        rez.append(key * p5_new[i, j])
        p5 = p5_new
    return sum(rez)


def part1_6(p, k, j):
    rez = []
    for m in range(1, k+1):
        sums = 0
        for mi in range(1, m):
            sums += rez[-mi] * np.linalg.matrix_power(p, mi)
        rez.append((np.linalg.matrix_power(p, m) - sums)[j, j])
    return rez[-1]

def part1_7(p, k, j):
    rez = []
    for m in range(1, k+1):
        sums = 0
        for mi in range(1, m):
            sums += rez[-mi] * np.linalg.matrix_power(p, mi)
        rez.append((np.linalg.matrix_power(p, m) - sums)[j, j])
    return sum(rez)

def part1_8(p, j):
    rez = []
    for m in range(1, 1000+1):
        sums = 0
        for mi in range(1, m):
            sums += rez[-mi] * np.linalg.matrix_power(p, mi)
        rez.append((np.linalg.matrix_power(p, m) - sums)[j, j])
    for i in range(len(rez)):
        rez[i] *= i+1
    return sum(rez)

def part1_9(p):
    matrix = p - np.eye(p.shape[0])
    matrix[-1, :] = 1
    vector = np.array([0] * (p.shape[0] - 1) + [1])
    return np.dot(np.linalg.inv(matrix), vector)

matrix = np.array([[0.05, 0, 0.51, 0, 0, 0, 0.59, 0, 0, 0, 0, 0, 0],
                   [0.39, 0.13, 0, 0, 0, 0.22, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0.26, 0.07, 0, 0, 0, 0, 0, 0.21, 0.3, 0, 0, 0],
                   [0, 0.13, 0.37, 0.04, 0.83, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0.25, 0, 0, 0.31, 0.17, 0.17, 0, 0, 0, 0, 0, 0, 0],
                   [0.15, 0.32, 0, 0, 0, 0.31, 0, 0.53, 0, 0.01, 0, 0, 0],
                   [0.16, 0.16, 0, 0, 0, 0.12, 0.13, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0.31, 0, 0.18, 0.07, 0.06, 0.17, 0, 0.36, 0, 0],
                   [0, 0, 0, 0.34, 0, 0, 0, 0.06, 0.03, 0.12, 0.55, 0.29, 0],
                   [0, 0, 0.05, 0, 0, 0, 0.21, 0, 0.18, 0.05, 0, 0, 0.43],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0.23, 0, 0.09, 0.17, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.39, 0, 0.47, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0.25, 0.18, 0.13, 0, 0.07, 0.57]])


print(f'1. Вероятность того, что за 10 шагов система перейдет из состояния 8 в состояние 10: {part1_1(matrix, 10, 7, 9)}\n')

a = np.array((0.03, 0.05, 0, 0.03, 0.16, 0.17, 0.05, 0.01, 0.06, 0.14, 0.13, 0.02, 0.1))
print(f'2. Вероятности состояний системы спустя 6 шагов, если в начальный момент вероятность состояний были следующими\n'
      f' A=(0,03;0,05;0;0,03;0,16;0,17;0,05;0,01;0,06;0,14;0,13;0,02;0,15)')
print(f'{part1_2(matrix, 6, a)} \n')

print(f'3. Вероятность первого перехода за 10 шагов из состояния 13 в состояние 5')
print(f'{part1_3(matrix, 10, 12, 4)}\n')

print(f'4. Вероятность перехода из состояния 5 в состояние 3 не позднее чем за 5 шагов')
print(f'{part1_4(matrix, 5, 4, 2)}\n')

print(f'5. Cреднее количество шагов для перехода из состояния 13 в состояние 11')
print(f'{part1_5(matrix, 12, 10)}\n')

print(f'6. Вероятность первого возвращения в состояние 6 за 5 шагов')
print(f'{part1_6(matrix, 5, 5)}\n')

print(f'7. Вероятность возвращения в состояние 6 не позднее чем за 7 шагов')
print(f'{part1_7(matrix, 7, 5)}\n')

print(f'8. Среднее время возвращения в состояние 1')
print(f'{part1_8(matrix, 0)}\n')

print(f'9. Установившиеся вероятности')
print(part1_9(matrix))
