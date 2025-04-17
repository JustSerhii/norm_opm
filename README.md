# Frobenius Norm with OpenMP

Цей проєкт реалізує три способи обчислення **норми Фробеніуса** для матриці заданого розміру:

- Послідовне обчислення
- Паралельне обчислення з OpenMP (`reduction`)
- Оптимізоване ручне розбиття з OpenMP (без атомарних операцій)

---

## Побудова проєкту

Використовується `CMake` для збірки проєкту:

```bash
cmake ..
cmake --build . --config Release
```
## Запуск програми
```bash
.\Release\frobenius_omp.exe <rows> <cols> <show_matrix (y/n)> <threads>
```
